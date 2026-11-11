# Python API and CSV format

## Query API

```python
from pathlib import Path
from litert_compat.matrix import Matrix

matrix = Matrix.load(Path("data/examples/matrix_example.json"))
verdict = matrix.lookup(
    "FULLY_CONNECTED",
    dtypes=["int8"],
    shape_meta={"rank": 2, "dynamic_shape": False},
)
# Verdict(status=..., matched_entry=..., reason=...,
#         backend="gpu_mldrift", litert_version="0.0.0-example")
```

Precedence, most-specific match wins: **dtypes AND constraints** beat **dtype-only** beat
**constraints-only** beat **op-only**. No match returns `status: "unknown"` — never a
guess, never a default to `delegated`. Constraint keys `max_rank`/`min_rank` compare
numerically against `shape_meta["rank"]`; other keys match by equality; missing
`shape_meta` information never satisfies a constraint. Ties at equal specificity with
contradictory statuses raise an error and are reported by `matrix validate`.

## CSV column contract

Column names map 1:1 to schema fields. Column order is free; header names are
case-insensitive; whitespace is trimmed. Multi-value cells use `;` separators. One rewrite
hint per row.

| Column | Maps to | Format |
|---|---|---|
| `op` | `entry.op` | required; exact TFLite builtin name, e.g. `FULLY_CONNECTED` |
| `dtypes` | `entry.dtypes` | `;`-separated, e.g. `float32;int8`; empty = op-generic entry |
| `constraints` | `entry.constraints` | `;`-separated `key=value`; values parsed as JSON scalars (`true`, `4`, `1.5`), else string |
| `status` | `entry.status` | required; `delegated \| partial \| fallback \| incorrect \| crash \| unknown` |
| `conditions` | `entry.conditions` | free text (quote the cell if it contains commas) |
| `rewrite_symptom` | `rewrite_hints[0].symptom` | free text; required if `rewrite` is set |
| `rewrite` | `rewrite_hints[0].rewrite` | free text; required if `rewrite_symptom` is set |
| `rewrite_expected_effect` | `rewrite_hints[0].expected_effect` | free text, optional |
| `provenance` | `entry.provenance` | required; `measured \| vendor_doc \| inferred \| example` |
| `evidence_source_model` | `evidence.source_model` | model id |
| `evidence_litert_version` | `evidence.litert_version` | version string |
| `evidence_date` | `evidence.date` | `YYYY-MM-DD` |

Snapshot-level fields (`litert_version`, `backend`, `target.*`, `generated_at`) are CLI
options, not CSV columns — one CSV import produces exactly one (backend × version)
snapshot.

`status: incorrect` means the delegate compiles and executes the op but produces numerics
outside tolerance vs the CPU reference — an observed, real failure mode and the worst one
for users.
