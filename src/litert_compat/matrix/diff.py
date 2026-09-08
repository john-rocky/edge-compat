"""Deterministic change summary between two matrix snapshots.

Semantic changes are entry additions/removals, status transitions, and
provenance transitions. Free-text edits (conditions, rewrite hints) do not
count — the exit-code contract exists so a release watcher can trigger on
real semantic change only.
"""

from __future__ import annotations

from typing import Any

from litert_compat.matrix.canonical import (
    Signature,
    entry_sort_key,
    format_signature,
    signature,
    signature_as_dict,
)


class BackendMismatchError(ValueError):
    """Diffing snapshots of different backends is meaningless (version-axis rule)."""


def _by_signature(doc: dict[str, Any]) -> dict[Signature, dict[str, Any]]:
    out: dict[Signature, dict[str, Any]] = {}
    for entry in sorted(doc.get("entries", []), key=entry_sort_key):
        out.setdefault(signature(entry), entry)
    return out


def _transition(sig: Signature, old: str, new: str) -> dict[str, Any]:
    return {**signature_as_dict(sig), "from": old, "to": new}


def diff_documents(
    old_doc: dict[str, Any], new_doc: dict[str, Any]
) -> dict[str, Any]:
    if old_doc["backend"] != new_doc["backend"]:
        raise BackendMismatchError(
            f"cannot diff across backends: {old_doc['backend']} vs {new_doc['backend']}"
        )

    old_map = _by_signature(old_doc)
    new_map = _by_signature(new_doc)
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    common = sorted(set(old_map) & set(new_map))

    status_transitions = [
        _transition(sig, old_map[sig]["status"], new_map[sig]["status"])
        for sig in common
        if old_map[sig]["status"] != new_map[sig]["status"]
    ]
    provenance_transitions = [
        _transition(sig, old_map[sig]["provenance"], new_map[sig]["provenance"])
        for sig in common
        if old_map[sig]["provenance"] != new_map[sig]["provenance"]
    ]

    old_ops = {sig[0] for sig in old_map}
    new_ops = {sig[0] for sig in new_map}

    result = {
        "backend": old_doc["backend"],
        "old": {
            "litert_version": old_doc["litert_version"],
            "generated_at": old_doc["generated_at"],
        },
        "new": {
            "litert_version": new_doc["litert_version"],
            "generated_at": new_doc["generated_at"],
        },
        "ops_added": sorted(new_ops - old_ops),
        "ops_removed": sorted(old_ops - new_ops),
        "entries_added": [
            {**signature_as_dict(sig), "status": new_map[sig]["status"]} for sig in added
        ],
        "entries_removed": [
            {**signature_as_dict(sig), "status": old_map[sig]["status"]} for sig in removed
        ],
        "status_transitions": status_transitions,
        "provenance_transitions": provenance_transitions,
    }
    result["has_changes"] = any(
        result[key]
        for key in ("entries_added", "entries_removed", "status_transitions",
                    "provenance_transitions")
    )
    return result


def _sig_of(item: dict[str, Any]) -> str:
    entry_like = {"op": item["op"], "dtypes": item["dtypes"], "constraints": item["constraints"]}
    return format_signature(signature(entry_like))


def render_text(result: dict[str, Any]) -> str:
    lines = [
        f"matrix diff — backend {result['backend']}: "
        f"{result['old']['litert_version']} ({result['old']['generated_at']}) → "
        f"{result['new']['litert_version']} ({result['new']['generated_at']})"
    ]
    if not result["has_changes"]:
        lines.append("no semantic differences")
        return "\n".join(lines)
    for item in result["entries_added"]:
        lines.append(f"  + {_sig_of(item)} (status={item['status']})")
    for item in result["entries_removed"]:
        lines.append(f"  - {_sig_of(item)} (status={item['status']})")
    for item in result["status_transitions"]:
        lines.append(f"  ~ {_sig_of(item)}: status {item['from']} → {item['to']}")
    for item in result["provenance_transitions"]:
        lines.append(f"  ~ {_sig_of(item)}: provenance {item['from']} → {item['to']}")
    return "\n".join(lines)
