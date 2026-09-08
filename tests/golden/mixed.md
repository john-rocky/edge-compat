# edge-lint report — `model_mixed_example.tflite`

**gpu_mldrift @ litert 0.0.0-example — 40.0% of 5 ops delegated · 1 partition · 3 blocking ops**

- model: `data/examples/model_mixed_example.tflite` (sha256 `9af4f555f71881924bded9d75b7e5c3febaa6c2672fbff6739cc9cf57f96f806`, 1 subgraph(s))
- matrix: `data/examples/matrix_example.json` (generated 2026-08-10)
- sync boundaries: 2

## Per-op verdicts

| loc | op | dtypes | status | claimed | provenance | reason |
|---|---|---|---|---|---|---|
| s0/n0 | FULLY_CONNECTED | int8 | fallback | no | example | matched entry specific to op, dtypes, and constraints |
| s0/n1 | CUSTOM | float32 | unknown | no | - | custom op 'ExampleCustomOp' — not a builtin, outside the matrix vocabulary |
| s0/n2 | CONV_2D | float32 | delegated | yes | example | matched entry specific to op and dtypes |
| s0/n3 | SOFTMAX | float16 | incorrect | yes | example | matched entry specific to op and dtypes |
| s0/n4 | TRANSPOSE_CONV | float32 | fallback | no | example | matched entry specific to op, dtypes, and constraints |

## Partition map

```
subgraph 0:
  (host)      node 0 FULLY_CONNECTED (fallback)
  (host)      node 1 CUSTOM (unknown)
  ----- host<->device sync -----
  [device #0] nodes 2-3 (2 ops: CONV_2D, SOFTMAX)
  ----- host<->device sync -----
  (host)      node 4 TRANSPOSE_CONV (fallback)
```

## Blocking ops (ranked)

> impact_score is a heuristic (output element count as a FLOPs proxy, adjacent-partition cost split across each CPU island) — a ranking aid, not a measurement

| rank | loc | op | status | impact |
|---|---|---|---|---|
| 1 | s0/n4 | TRANSPOSE_CONV | fallback | 128.0 |
| 2 | s0/n0 | FULLY_CONNECTED | fallback | 64.0 |
| 3 | s0/n1 | CUSTOM | unknown | 64.0 |

## Rewrite suggestions

_Verbatim from the matrix, never generated._

- **s0/n0 FULLY_CONNECTED**
  - symptom: partition split at FULLY_CONNECTED with dynamic batch
  - rewrite: export with a fixed batch dimension
  - expected effect: single contiguous GPU partition

## Custom ops

- s0/n1 CUSTOM `ExampleCustomOp` (outside the matrix vocabulary; not probeable)

## Next steps

- fallback / incorrect / crash: apply the rewrite hint recorded above, or try the recorded transforms with `edge-fix run <model> --matrix <same snapshot> --rules data/transforms` (a dry run unless --apply is given).
- unknown / needs probe: no measured entry for gpu_mldrift @ litert 0.0.0-example — generate probe fixtures with `edge-compat probe gen` and measure them on your device, or look for a newer snapshot at https://john-rocky.github.io/edge-compat/data/matrix/
- verdicts hold for gpu_mldrift @ litert 0.0.0-example only; delegate support changes between runtime versions.
