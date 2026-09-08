# edge-lint report — `model_clean_example.tflite`

**gpu_mldrift @ litert 0.0.0-example — 100.0% of 3 ops delegated · 1 partition · 0 blocking ops**

- model: `data/examples/model_clean_example.tflite` (sha256 `e763326d958dca53e16643c11ce300fbac480c475cfd751b6d8336650089670f`, 1 subgraph(s))
- matrix: `data/examples/matrix_example.json` (generated 2026-08-10)
- sync boundaries: 0

## Per-op verdicts

| loc | op | dtypes | status | claimed | provenance | reason |
|---|---|---|---|---|---|---|
| s0/n0 | CONV_2D | float32 | delegated | yes | example | matched entry specific to op and dtypes |
| s0/n1 | AVERAGE_POOL_2D | float32 | delegated | yes | example | matched entry specific to op and dtypes |
| s0/n2 | CONV_2D | float32 | delegated | yes | example | matched entry specific to op and dtypes |

## Partition map

```
subgraph 0:
  [device #0] nodes 0-2 (3 ops: CONV_2D, AVERAGE_POOL_2D, CONV_2D)
```
