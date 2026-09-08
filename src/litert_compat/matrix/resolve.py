"""Backend -> snapshot resolution over a snapshots directory (Phase 14).

Policy (spec 14.1): exactly one snapshot for the backend -> that one; multiple
versions -> the highest numeric version unless pinned; anything ambiguous ->
error, never a coin flip (the playground's refusal pattern). Snapshot identity
comes from each file's own `backend` / `litert_version` fields — file names
are never trusted (Integration Rules §D: one file per backend x version).

Discovery is strict, matching the sweep/device-run pattern: every `*.json`
in the directory must be a readable snapshot header; anything else is an
error naming the file, not a silent skip.
"""

from __future__ import annotations

from pathlib import Path

from litert_compat.matrix.canonical import load_json


class SnapshotResolutionError(ValueError):
    """The snapshots directory cannot serve the request unambiguously."""


def numeric_version(version: str) -> tuple[int, ...] | None:
    """`x.y.z` as an orderable int tuple, or None when not purely numeric."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


def _discover(snapshots_dir: Path) -> list[tuple[str, str, Path]]:
    """-> sorted [(backend, litert_version, path)] for every snapshot file."""
    if not snapshots_dir.is_dir():
        raise SnapshotResolutionError(f"snapshots directory {snapshots_dir} does not exist")
    found: list[tuple[str, str, Path]] = []
    for path in sorted(snapshots_dir.glob("*.json")):
        try:
            doc = load_json(path)
        except ValueError as exc:
            raise SnapshotResolutionError(f"{path}: not valid JSON: {exc}") from exc
        backend = doc.get("backend") if isinstance(doc, dict) else None
        version = doc.get("litert_version") if isinstance(doc, dict) else None
        if not isinstance(backend, str) or not isinstance(version, str):
            raise SnapshotResolutionError(
                f"{path}: not a matrix snapshot (missing string `backend` / "
                "`litert_version` fields)"
            )
        found.append((backend, version, path))
    return found


def resolve_snapshot(
    snapshots_dir: Path,
    backend: str,
    litert_version: str | None = None,
) -> Path:
    """Path of the snapshot under `snapshots_dir` serving `backend`.

    - Exactly one snapshot for the backend: that one.
    - Multiple versions: the highest numeric version — unless `litert_version`
      pins an exact one.
    - Ambiguity (duplicate backend x version, or unpinned non-numeric
      versions): SnapshotResolutionError, never a coin flip.
    """
    snapshots = _discover(snapshots_dir)
    candidates = [(version, path) for b, version, path in snapshots if b == backend]
    if not candidates:
        available = sorted({b for b, _, _ in snapshots})
        raise SnapshotResolutionError(
            f"no matrix snapshot for backend {backend!r} under {snapshots_dir} "
            f"(available backends: {', '.join(available) if available else 'none'})"
        )

    versions = [version for version, _ in candidates]
    duplicates = sorted({v for v in versions if versions.count(v) > 1})
    if duplicates:
        raise SnapshotResolutionError(
            f"duplicate snapshots for backend {backend!r} at litert_version "
            f"{', '.join(duplicates)} under {snapshots_dir}; one file per "
            "(backend x litert_version)"
        )

    if litert_version is not None:
        for version, path in candidates:
            if version == litert_version:
                return path
        raise SnapshotResolutionError(
            f"no snapshot for backend {backend!r} at litert_version "
            f"{litert_version!r} under {snapshots_dir} "
            f"(available versions: {', '.join(sorted(versions))})"
        )

    if len(candidates) == 1:
        return candidates[0][1]

    non_numeric = sorted(v for v in versions if numeric_version(v) is None)
    if non_numeric:
        raise SnapshotResolutionError(
            f"multiple snapshots for backend {backend!r} include non-numeric "
            f"litert_version {', '.join(non_numeric)} under {snapshots_dir}; "
            "versions cannot be ordered — pin litert_version explicitly"
        )
    return max(candidates, key=lambda item: numeric_version(item[0]) or ())[1]
