"""Probe signatures: the complete lookup signature a probe fixture exercises.

Two sources, both spec-fixed: linter `needs_probe` records (already complete
signatures — §G.3) and `inferred` entries selected out of a matrix snapshot
(their `constraints` map deterministically onto a concrete probe shape).
A signature that cannot be realized as a fixture is *unprobeable* and is
surfaced as remaining work — never silently dropped, never guessed around.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

DEFAULT_PROBE_RANK = 4

# Constraint keys the generator can realize in a fixture. Anything else
# (keep_dims, half_pixel_centers, ...) needs builtin-option extraction, which
# classification does not do yet (Phase 2 deferred item) — unprobeable.
_REALIZABLE_CONSTRAINTS = frozenset({"dynamic_shape", "rank", "max_rank", "min_rank"})


class UnprobeableSignatureError(ValueError):
    """The signature cannot be realized as a single-op probe fixture."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _meta_key(shape_meta: dict[str, Any]) -> str:
    return json.dumps(shape_meta, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProbeSignature:
    """One (backend, op, dtypes, shape_meta) lookup signature to probe."""

    backend: str
    op: str
    dtypes: tuple[str, ...]
    shape_meta_key: str  # canonical JSON — keeps the dataclass hashable

    @property
    def shape_meta(self) -> dict[str, Any]:
        return json.loads(self.shape_meta_key)

    @property
    def fixture_id(self) -> str:
        """Stable fixture id. Backend-independent: the same signature probes
        identically on every backend; only the runner differs."""
        digest = hashlib.sha256(self.shape_meta_key.encode("utf-8")).hexdigest()[:8]
        dtypes = "-".join(self.dtypes) or "untyped"
        return f"{self.op}__{dtypes}__{digest}"

    def sort_key(self) -> tuple[str, str, tuple[str, ...], str]:
        return (self.backend, self.op, self.dtypes, self.shape_meta_key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "op": self.op,
            "dtypes": list(self.dtypes),
            "shape_meta": self.shape_meta,
        }


def make_signature(
    backend: str, op: str, dtypes: list[str] | tuple[str, ...], shape_meta: dict[str, Any]
) -> ProbeSignature:
    return ProbeSignature(
        backend=backend,
        op=op,
        dtypes=tuple(dtypes),
        shape_meta_key=_meta_key(shape_meta),
    )


def signatures_from_lint_report(doc: dict[str, Any]) -> list[ProbeSignature]:
    """Extract `needs_probe` signatures from a lint report object."""
    records = doc.get("needs_probe")
    if not isinstance(records, list):
        raise ValueError("not a lint report: missing needs_probe list")
    out = []
    for record in records:
        out.append(
            make_signature(
                backend=record["backend"],
                op=record["op"],
                dtypes=record["dtypes"],
                shape_meta=record["shape_meta"],
            )
        )
    return sorted(out, key=ProbeSignature.sort_key)


def shape_meta_from_constraints(constraints: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic mapping from entry constraints to a concrete probe shape_meta.

    rank precedence: explicit `rank` > `max_rank` (exercise the boundary) >
    `min_rank` > DEFAULT_PROBE_RANK. Raises UnprobeableSignatureError for
    constraint keys the generator cannot realize.
    """
    constraints = constraints or {}
    for key in sorted(constraints):
        if key not in _REALIZABLE_CONSTRAINTS:
            raise UnprobeableSignatureError(
                f"constraint {key!r} cannot be realized by the probe generator"
            )
    for key in ("rank", "max_rank", "min_rank"):
        if key in constraints:
            rank = constraints[key]
            break
    else:
        rank = DEFAULT_PROBE_RANK
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise UnprobeableSignatureError(f"rank constraint {rank!r} is not a valid rank")
    return {
        "dynamic_shape": bool(constraints.get("dynamic_shape", False)),
        "rank": rank,
    }


def signature_from_entry(backend: str, entry: dict[str, Any]) -> ProbeSignature:
    """Probe signature for a matrix entry (raises if its constraints cannot
    be realized). The entry keeps its own identity; this signature only
    determines the fixture."""
    return make_signature(
        backend=backend,
        op=entry["op"],
        dtypes=entry.get("dtypes") or [],
        shape_meta=shape_meta_from_constraints(entry.get("constraints")),
    )
