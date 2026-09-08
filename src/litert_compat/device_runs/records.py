"""Device-run record validation, snapshot discovery, and append-only writes.

Snapshot layout (spec 13.3, mirroring the sweep for freshness reuse):
`<root>/<runtime_version>/<YYYY-MM-DD>/<model_id>__<device>.json`. The
version segment is the RUNTIME version axis — litert for the .tflite-on-NPU
lane, litert-lm for the .litertlm lane — with the runtime named inside every
record's env. A snapshot directory is homogeneous: every record carries the
same env.runtime, and env.runtime_version equals the directory's version
segment, so a snapshot can never lie about its own version axis.

Append-only discipline: records are added, never overwritten — re-measurement
lands in a new dated snapshot, exactly like the sweep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litert_compat.cards.schema_io import schema_errors
from litert_compat.freshness.releases import parse_version
from litert_compat.matrix.canonical import load_json, write_canonical

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DEVICE_RUN_SCHEMA = "device_run_result.schema.json"


class DeviceRunError(ValueError):
    """A device-run document, snapshot directory, or write request is invalid."""


def device_run_errors(doc: Any) -> list[str]:
    """Schema validation errors for one device-run document; empty = valid."""
    return schema_errors(doc, DEVICE_RUN_SCHEMA)


def record_condition(record: dict[str, Any]) -> float | None:
    """The prompt-length condition of one accelerator record:
    `metrics.prefill_tokens`. Throughput depends on it (the default 19-token
    prompt is a latency floor that understates prefill ~12.7x against a
    205-token prompt), so together with `accelerator` it is the record's
    identity. None for records measured before the condition dimension
    existed (DECISIONS #162) and for runs whose source states no count."""
    metrics = record.get("metrics") or {}
    return metrics.get("prefill_tokens")


def record_signature(record: dict[str, Any]) -> str | None:
    """The signature (subgraph) a CV-lane record measured, for multi-signature
    .tflite files whose signatures are different measurements (decode vs
    prefill_128, encode vs decode_1). None for single/default-signature files
    and for every record written before the field existed."""
    return record.get("signature")


def record_label(record: dict[str, Any]) -> str:
    """Human-readable identity of one accelerator record: the accelerator id,
    with the prompt-length condition when the record carries one — 'gpu' /
    'gpu@205tok' — or the measured signature for a multi-signature file —
    'cpu_xnnpack@decode'. Used as the dedup key wherever records from
    different measurements meet (append-only writes, card enrichment, deltas)."""
    condition = record_condition(record)
    if condition is not None:
        return f"{record['accelerator']}@{condition:g}tok"
    signature = record_signature(record)
    if signature is not None:
        return f"{record['accelerator']}@{signature}"
    return record["accelerator"]


def record_sort_key(record: dict[str, Any]) -> tuple[str, bool, float, str]:
    """Deterministic in-file order: accelerator id, then unconditioned before
    conditioned, then ascending prompt length, then signature name."""
    condition = record_condition(record)
    return (
        record["accelerator"],
        condition is not None,
        condition or 0.0,
        record_signature(record) or "",
    )


def device_status(record: dict[str, Any]) -> str:
    """Deterministic one-word summary of one accelerator record — the same
    vocabulary and derivation as `litert_compat.cards.index.browser_status`
    (DECISIONS #62), so the site never grows a second status dialect."""
    if not record["loads"]:
        return "load_failed"
    if not record["runs"]:
        return "run_failed"
    if record["output_match"] is False:
        return "output_mismatch"
    if record["full_delegation"] is False:
        return "fallback"
    return "pass"


@dataclass(frozen=True)
class DeviceRunSnapshot:
    """One dated batch of device-run results against one runtime version."""

    runtime: str
    runtime_version: str
    date: str
    path: Path
    #: (model_id, device) -> parsed device_run_result document.
    docs: dict[tuple[str, str], dict[str, Any]] = field(compare=False)

    def sort_key(self) -> tuple[tuple[int, ...], str, str]:
        return (parse_version(self.runtime_version) or (), self.runtime_version, self.date)


def _load_snapshot_dir(version: str, date: str, path: Path) -> DeviceRunSnapshot:
    docs: dict[tuple[str, str], dict[str, Any]] = {}
    runtimes: set[str] = set()
    for file in sorted(path.glob("*.json")):
        try:
            doc = load_json(file)
        except (OSError, ValueError) as exc:
            raise DeviceRunError(f"{file}: not valid JSON: {exc}") from exc
        errors = device_run_errors(doc)
        if errors:
            details = "; ".join(errors[:3])
            raise DeviceRunError(f"{file}: not a valid device-run result: {details}")
        key = (doc["model_id"], doc["device"])
        if file.name != f"{key[0]}__{key[1]}.json":
            raise DeviceRunError(
                f"{file}: file name must be <model_id>__<device>.json "
                f"(document says {key[0]!r} on {key[1]!r})"
            )
        for record in doc["results"]:
            env = record["env"]
            runtimes.add(env["runtime"])
            if env["runtime_version"] != version:
                raise DeviceRunError(
                    f"{file}: record env.runtime_version {env['runtime_version']!r} "
                    f"does not match the snapshot directory version {version!r} — "
                    "a snapshot must not lie about its own version axis"
                )
        docs[key] = doc
    if len(runtimes) > 1:
        raise DeviceRunError(
            f"{path}: snapshot mixes runtimes {sorted(runtimes)} — one snapshot "
            "directory covers one runtime (litert and litert-lm share a version "
            "string only by accident; split the batch)"
        )
    runtime = next(iter(runtimes)) if runtimes else "litert"
    return DeviceRunSnapshot(
        runtime=runtime, runtime_version=version, date=date, path=path, docs=docs
    )


def discover_device_run_snapshots(root: Path) -> list[DeviceRunSnapshot]:
    """All snapshots under `root`, ordered oldest -> newest by (numeric
    version, version string, date). Strict discovery, mirroring the sweep:
    a malformed snapshot is an error, never silently skipped."""
    if not root.is_dir():
        return []
    snapshots: list[DeviceRunSnapshot] = []
    for version_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        date_dirs = sorted(p for p in version_dir.iterdir() if p.is_dir())
        if not date_dirs:
            raise DeviceRunError(
                f"{version_dir}: snapshot version directory has no <YYYY-MM-DD> subdirectory"
            )
        for date_dir in date_dirs:
            if not _DATE_RE.match(date_dir.name):
                raise DeviceRunError(
                    f"{date_dir}: snapshot date directory must be named YYYY-MM-DD"
                )
            snapshots.append(_load_snapshot_dir(version_dir.name, date_dir.name, date_dir))
    return sorted(snapshots, key=DeviceRunSnapshot.sort_key)


def snapshots_by_runtime(
    snapshots: list[DeviceRunSnapshot],
) -> dict[str, list[DeviceRunSnapshot]]:
    """Group ordered snapshots by runtime, preserving order — deltas compare
    litert against litert and litert-lm against litert-lm, never across."""
    groups: dict[str, list[DeviceRunSnapshot]] = {}
    for snapshot in snapshots:
        groups.setdefault(snapshot.runtime, []).append(snapshot)
    return dict(sorted(groups.items()))


def write_device_run(root: Path, doc: dict[str, Any], *, date: str) -> Path:
    """Write `doc` into the snapshot at `<root>/<runtime_version>/<date>/`,
    merging into an existing <model_id>__<device>.json by appending records.

    Append-only: a record whose identity — (accelerator × prompt-length
    condition, see `record_label`) — already exists in the target file is
    refused, never overwritten; a re-measurement belongs in a new dated
    snapshot. The same accelerator under a different `metrics.prefill_tokens`
    is a different fact and merges in. The written file always validates
    against the schema."""
    errors = device_run_errors(doc)
    if errors:
        details = "\n".join(f"  {e}" for e in errors)
        raise DeviceRunError(f"refusing to write an invalid device-run document:\n{details}")
    if not _DATE_RE.match(date):
        raise DeviceRunError(f"snapshot date must be YYYY-MM-DD, got {date!r}")
    versions = {r["env"]["runtime_version"] for r in doc["results"]}
    runtimes = {r["env"]["runtime"] for r in doc["results"]}
    if len(versions) != 1 or len(runtimes) != 1:
        raise DeviceRunError(
            f"document mixes runtimes {sorted(runtimes)} / versions {sorted(versions)}: "
            "one write targets one (runtime x version) snapshot"
        )
    version = next(iter(versions))
    target = root / version / date / f"{doc['model_id']}__{doc['device']}.json"
    merged = doc
    if target.is_file():
        existing = load_json(target)
        for key in ("model_id", "device", "artifact", "quantization"):
            if existing.get(key) != doc.get(key):
                raise DeviceRunError(
                    f"{target}: existing file disagrees on {key!r} "
                    f"({existing.get(key)!r} != {doc.get(key)!r})"
                )
        old_sha, new_sha = existing.get("artifact_sha256"), doc.get("artifact_sha256")
        if old_sha is not None and new_sha is not None and old_sha != new_sha:
            raise DeviceRunError(
                f"{target}: existing file measured artifact_sha256 {old_sha} but the "
                f"incoming record names {new_sha} — a different artifact belongs in its "
                "own record, not merged under the same name"
            )
        taken = {record_label(r) for r in existing["results"]}
        added = {record_label(r) for r in doc["results"]}
        overlap = sorted(taken & added)
        if overlap:
            if all(r in existing["results"] for r in doc["results"]):
                # Idempotent re-ingest: every incoming record is identical to
                # one already present — the same fact re-run is a no-op, so
                # source reports can be safely re-ingested after a partial run.
                return target
            raise DeviceRunError(
                f"{target}: accelerator record(s) {', '.join(overlap)} already exist — "
                "snapshots are append-only; a re-measurement belongs in a new dated snapshot"
            )
        merged = dict(existing)
        if old_sha is None and new_sha is not None:
            merged["artifact_sha256"] = new_sha
        merged["results"] = sorted(
            [*existing["results"], *doc["results"]], key=record_sort_key
        )
        errors = device_run_errors(merged)
        if errors:
            details = "\n".join(f"  {e}" for e in errors)
            raise DeviceRunError(f"merged document would be invalid:\n{details}")
    else:
        merged = dict(doc)
        merged["results"] = sorted(doc["results"], key=record_sort_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_canonical(merged, target)
    return target


def select_device_runs(
    snapshots: list[DeviceRunSnapshot],
) -> list[tuple[Path, dict[str, Any]]]:
    """Cell-wise newest selection over every snapshot given (DECISIONS #165):
    for each (model_id x device x `record_label`) cell, the record from the
    newest snapshot that measured it — "newest" in this module's snapshot
    order: numeric runtime version, then date, with non-numeric versions
    (`gallery-*`, `selfbuilt-*`) before every numeric one. Runtime version is
    not part of the cell, so a newer measurement of the same cell supersedes
    the older one, and a cell measured only on an older runtime stays
    selected: the staleness axis flags it, the card does not hide it.

    Returns (file, document) pairs in the shape `cards.enrich.load_device_runs`
    returns, each document's `results` cut to the records that won their cell
    (a file whose every record was superseded does not appear), sorted by
    file. A cell measured under two runtimes (litert and litert-lm) is
    refused: their version axes are not comparable, so it has no newest record.
    """
    winners: dict[tuple[str, str, str], tuple[str, Path, dict[str, Any], dict[str, Any]]] = {}
    errors: list[str] = []
    for snapshot in sorted(snapshots, key=DeviceRunSnapshot.sort_key):
        for (model_id, device), doc in sorted(snapshot.docs.items()):
            path = snapshot.path / f"{model_id}__{device}.json"
            for record in doc["results"]:
                key = (model_id, device, record_label(record))
                prior = winners.get(key)
                if prior is not None and prior[0] != snapshot.runtime:
                    errors.append(
                        f"{path}: {model_id} {device} {key[2]} is also measured under "
                        f"{prior[0]} in {prior[1]} — the {prior[0]} and {snapshot.runtime} "
                        "version axes are not comparable, so the cell has no newest record"
                    )
                    continue
                winners[key] = (snapshot.runtime, path, doc, record)
    if errors:
        raise DeviceRunError("\n".join(errors))
    by_file: dict[Path, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for _runtime, path, doc, record in winners.values():
        by_file.setdefault(path, (doc, []))[1].append(record)
    return [
        (path, {**doc, "results": sorted(records, key=record_sort_key)})
        for path, (doc, records) in sorted(by_file.items())
    ]
