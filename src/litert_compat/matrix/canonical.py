"""Deterministic (canonical) serialization and entry identity for matrix snapshots.

Every writer in this package emits canonical form: sorted object keys, entries
sorted by signature, 2-space indent, trailing newline. Same input, same bytes —
so diffs stay reviewable and generated files are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Signature = tuple[str, tuple[str, ...], str]


def canonical_dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_canonical(doc: Any, path: Path) -> None:
    path.write_text(canonical_dumps(doc), encoding="utf-8")


def constraints_key(constraints: dict[str, Any] | None) -> str:
    return json.dumps(constraints or {}, sort_keys=True, separators=(",", ":"))


def signature(entry: dict[str, Any]) -> Signature:
    """Identity of an entry: (op, sorted dtypes, canonical constraints)."""
    return (
        entry.get("op", ""),
        tuple(sorted(entry.get("dtypes") or [])),
        constraints_key(entry.get("constraints")),
    )


def signature_as_dict(sig: Signature) -> dict[str, Any]:
    return {
        "op": sig[0],
        "dtypes": list(sig[1]),
        "constraints": json.loads(sig[2]),
    }


def format_signature(sig: Signature) -> str:
    dtypes = ",".join(sig[1]) if sig[1] else "*"
    return f"{sig[0]} [{dtypes}] {sig[2]}"


def entry_sort_key(entry: dict[str, Any]) -> tuple[str, tuple[str, ...], str, str, str]:
    sig = signature(entry)
    return (*sig, entry.get("status", ""), entry.get("provenance", ""))


def sort_entries(doc: dict[str, Any]) -> None:
    doc["entries"] = sorted(doc.get("entries", []), key=entry_sort_key)
