# edge-lint report — `model_split_example.tflite`

**gpu_mldrift @ litert 0.0.0-example — 40.0% of 5 ops delegated · 2 partitions · 3 blocking ops**

- model: `data/examples/model_split_example.tflite` (sha256 `d6039cd535890604b0a4ae0552cfb970de15cf3fbfe33694442f3fad02424f6e`, 1 subgraph(s))
- matrix: `data/examples/matrix_example.json` (generated 2026-08-10)
- sync boundaries: 3

## Per-op verdicts

| loc | op | dtypes | status | claimed | provenance | reason |
|---|---|---|---|---|---|---|
| s0/n0 | CONV_2D | float32 | delegated | yes | example | matched entry specific to op and dtypes |
| s0/n1 | GATHER_ND | float32 | fallback | no | example | matched entry specific to op and dtypes |
| s0/n2 | RESHAPE | float32 | unknown | no | - | no entry for op RESHAPE |
| s0/n3 | FULLY_CONNECTED | float32 | delegated | yes | example | matched entry specific to op, dtypes, and constraints |
| s0/n4 | SOFTMAX | float32 | unknown | no | - | entries exist for op SOFTMAX but none match dtypes=['float32'] shape_meta={'dynamic_shape': False, 'rank': 2} |

## Partition map

```
subgraph 0:
  [device #0] nodes 0-0 (1 op: CONV_2D)
  ----- host<->device sync -----
  (host)      node 1 GATHER_ND (fallback)
  (host)      node 2 RESHAPE (unknown)
  ----- host<->device sync -----
  [device #1] nodes 3-3 (1 op: FULLY_CONNECTED)
  ----- host<->device sync -----
  (host)      node 4 SOFTMAX (unknown)
```

## Blocking ops (ranked)

> impact_score is a heuristic (output element count as a FLOPs proxy, adjacent-partition cost split across each CPU island) — a ranking aid, not a measurement

| rank | loc | op | status | impact |
|---|---|---|---|---|
| 1 | s0/n1 | GATHER_ND | fallback | 1029.0 |
| 2 | s0/n2 | RESHAPE | unknown | 1029.0 |
| 3 | s0/n4 | SOFTMAX | unknown | 10.0 |

## Rewrite suggestions

_Verbatim from the matrix, never generated._

- **s0/n1 GATHER_ND**
  - symptom: graph splits at GATHER_ND
  - rewrite: rewrite step>1 slicing as gather over a precomputed index tensor
  - expected effect: removes the CPU fallback island

## Needs probe

Complete lookup signatures — sufficient to construct a probe without reopening the model.

| op | dtypes | shape_meta | backend |
|---|---|---|---|
| RESHAPE | float32 | `{"dynamic_shape":false,"rank":2}` | gpu_mldrift |
| SOFTMAX | float32 | `{"dynamic_shape":false,"rank":2}` | gpu_mldrift |

## Next steps

- fallback / incorrect / crash: apply the rewrite hint recorded above, or try the recorded transforms with `edge-fix run <model> --matrix <same snapshot> --rules data/transforms` (a dry run unless --apply is given).
- unknown / needs probe: no measured entry for gpu_mldrift @ litert 0.0.0-example — generate probe fixtures with `edge-compat probe gen` and measure them on your device, or look for a newer snapshot at https://john-rocky.github.io/edge-compat/data/matrix/
- verdicts hold for gpu_mldrift @ litert 0.0.0-example only; delegate support changes between runtime versions.
