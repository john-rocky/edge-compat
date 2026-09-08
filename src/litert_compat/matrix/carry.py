"""Carry a snapshot forward to a new LiteRT version — never in place.

Every `measured` entry downgrades to `provenance: "inferred"` with
`evidence.derived_from` pinning the source snapshot's litert_version and
generated_at. Re-measurement later upgrades entries back to `measured`.
The `inferred` count of the resulting file IS the staleness debt.
"""

from __future__ import annotations

import copy
from typing import Any

from litert_compat.matrix.canonical import sort_entries


def carry_forward(
    doc: dict[str, Any], *, to_litert_version: str, generated_at: str
) -> dict[str, Any]:
    if to_litert_version == doc["litert_version"]:
        raise ValueError(
            f"target litert_version {to_litert_version} equals the source snapshot's; "
            "carry-forward must produce a new version's snapshot"
        )

    source_version = doc["litert_version"]
    source_date = doc["generated_at"]

    new_doc = copy.deepcopy(doc)
    new_doc["litert_version"] = to_litert_version
    new_doc["generated_at"] = generated_at
    for entry in new_doc.get("entries", []):
        if entry.get("provenance") == "measured":
            entry["provenance"] = "inferred"
            evidence = entry.setdefault("evidence", {})
            evidence["derived_from"] = {
                "litert_version": source_version,
                "date": source_date,
            }
        # Already-inferred entries keep their existing derived_from: it pins the
        # original measurement, which is the honest ancestry.
    sort_entries(new_doc)
    return new_doc
