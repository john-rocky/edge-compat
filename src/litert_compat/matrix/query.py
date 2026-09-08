"""Query API: `Matrix.lookup(op, dtypes, shape_meta) -> Verdict`.

Precedence: most-specific match wins — dtypes AND constraints (3) beat
dtype-only (2) beat constraints-only (1) beat op-only (0). No match returns
status "unknown" — never a guess, never a default to "delegated". Ties at equal
specificity with contradictory statuses raise; `matrix validate` flags them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_compat.matrix.canonical import entry_sort_key, load_json
from litert_compat.matrix.validate import MatrixValidationError, validate_document


@dataclass(frozen=True)
class Verdict:
    """Lookup result. Always carries the snapshot's backend and litert_version
    so no output can present one backend's result as another's."""

    status: str
    matched_entry: dict[str, Any] | None
    reason: str
    backend: str
    litert_version: str


class MatrixLookupTieError(ValueError):
    """Two entries matched at equal specificity with contradictory statuses.

    This is a data defect: run `edge-compat matrix validate` on the file.
    """


def dtype_match(
    entry_dtypes: list[str] | None, query_dtypes: list[str] | None
) -> tuple[bool, bool]:
    """-> (matches, dtype_specific). An entry without dtypes is op-generic.
    Public: the fix engine reuses the exact lookup semantics for rule matching."""
    if not entry_dtypes:
        return True, False
    if not query_dtypes:
        # Entry is restricted to specific dtypes and the query names none:
        # we cannot confirm the restriction, so the entry does not match.
        return False, True
    return set(query_dtypes) <= set(entry_dtypes), True


def constraints_match(
    entry_constraints: dict[str, Any] | None, shape_meta: dict[str, Any] | None
) -> tuple[bool, bool]:
    """-> (matches, constraint_specific). Missing shape_meta info never
    satisfies a constraint — unknowable must not resolve toward a verdict.
    Public: the fix engine reuses the exact lookup semantics for rule matching."""
    if not entry_constraints:
        return True, False
    meta = shape_meta or {}
    for key, expected in entry_constraints.items():
        if key in ("max_rank", "min_rank"):
            rank = meta.get("rank")
            if not isinstance(rank, int | float) or isinstance(rank, bool):
                return False, True
            if key == "max_rank" and rank > expected:
                return False, True
            if key == "min_rank" and rank < expected:
                return False, True
        elif key not in meta or meta[key] != expected:
            return False, True
    return True, True


_REASONS = {
    3: "matched entry specific to op, dtypes, and constraints",
    2: "matched entry specific to op and dtypes",
    1: "matched entry specific to op and constraints",
    0: "matched op-level entry",
}


class Matrix:
    """A loaded, validated (backend x litert_version) compatibility snapshot."""

    def __init__(self, doc: dict[str, Any], source: Path | None = None) -> None:
        findings = validate_document(doc)
        if findings:
            raise MatrixValidationError(str(source) if source else "<memory>", findings)
        self.doc = doc
        self.source = source
        self.backend: str = doc["backend"]
        self.litert_version: str = doc["litert_version"]
        self._by_op: dict[str, list[dict[str, Any]]] = {}
        for entry in sorted(doc["entries"], key=entry_sort_key):
            self._by_op.setdefault(entry["op"], []).append(entry)

    @classmethod
    def load(cls, path: Path) -> Matrix:
        return cls(load_json(path), source=path)

    def _unknown(self, reason: str) -> Verdict:
        return Verdict(
            status="unknown",
            matched_entry=None,
            reason=reason,
            backend=self.backend,
            litert_version=self.litert_version,
        )

    def lookup(
        self,
        op: str,
        dtypes: list[str] | None = None,
        shape_meta: dict[str, Any] | None = None,
        operand_meta: dict[str, Any] | None = None,
    ) -> Verdict:
        """`operand_meta` (operand_a..d = constant|activation) widens what
        constraints can match without entering the reported shape_meta or the
        reason strings — lint reports stay pinned."""
        candidates = self._by_op.get(op)
        if not candidates:
            return self._unknown(f"no entry for op {op}")

        match_meta = {**(shape_meta or {}), **(operand_meta or {})}
        matches: list[tuple[int, dict[str, Any]]] = []
        for entry in candidates:
            dtype_ok, dtype_specific = dtype_match(entry.get("dtypes"), dtypes)
            constraints_ok, constraint_specific = constraints_match(
                entry.get("constraints"), match_meta
            )
            if dtype_ok and constraints_ok:
                score = (2 if dtype_specific else 0) + (1 if constraint_specific else 0)
                matches.append((score, entry))

        if not matches:
            return self._unknown(
                f"entries exist for op {op} but none match dtypes={dtypes} "
                f"shape_meta={shape_meta}"
            )

        top = max(score for score, _ in matches)
        best = [entry for score, entry in matches if score == top]
        if len({entry["status"] for entry in best}) > 1:
            raise MatrixLookupTieError(
                f"contradictory entries tie at equal specificity for op {op} "
                f"(dtypes={dtypes}, shape_meta={shape_meta}); "
                "run `edge-compat matrix validate` on this file"
            )
        chosen = best[0]  # candidates are pre-sorted canonically; first is deterministic
        return Verdict(
            status=chosen["status"],
            matched_entry=chosen,
            reason=_REASONS[top],
            backend=self.backend,
            litert_version=self.litert_version,
        )

    def rewrite_hints(
        self, op: str, condition: str | None = None
    ) -> list[dict[str, Any]]:
        """Rewrite hints for `op`, verbatim from this snapshot's entries.

        Hints are curated data — passed through unchanged, never generated;
        no match returns an empty list. `condition` narrows the result by
        case-insensitive substring against the owning entry's `conditions`
        text and the hint's own symptom/rewrite/expected_effect text. Entries
        are pre-sorted canonically, so ordering is deterministic.
        """
        needle = condition.casefold() if condition else None
        hints: list[dict[str, Any]] = []
        for entry in self._by_op.get(op, []):
            for hint in entry.get("rewrite_hints") or []:
                if needle is not None:
                    haystack = " ".join(
                        str(text)
                        for text in (
                            entry.get("conditions"),
                            hint.get("symptom"),
                            hint.get("rewrite"),
                            hint.get("expected_effect"),
                        )
                        if text
                    ).casefold()
                    if needle not in haystack:
                        continue
                hints.append(hint)
        return hints
