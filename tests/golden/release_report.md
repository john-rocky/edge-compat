# edge-compat release check — 1.3.0 (2026-08-10)

Semantic changes: **yes**. Numeric tolerance vs CPU reference: abs 1e-05, rel 0.001.

## Runners

- `fake`: available

## gpu_mldrift: 1.2.0 → 1.3.0

- snapshot: `gpu_mldrift__1.2.0.json` → `gpu_mldrift__1.3.0.json`
- runner: `fake`
- remaining `inferred` entries (staleness debt): 5
- probe outcomes: conflict=1, created=2, runner_error=1, unknown_result=1, unprobeable=2, upgraded=2

| op | dtypes | constraints | kind | disposition | status | max_abs_diff | note |
|---|---|---|---|---|---|---|---|
| ABS | float32 | `{}` | entry | runner_error | - | - | device fell off the desk |
| LSTM | float32 | `{}` | entry | unprobeable | - | - | no probe template for op LSTM |
| MEAN | float32 | `{"keep_dims":true}` | entry | unprobeable | - | - | constraint 'keep_dims' cannot be realized by the probe generator |
| NEG | float32 | `{}` | entry | upgraded | delegated | 0.000e+00 | - |
| RESHAPE | float32 | `{"dynamic_shape":false,"rank":2}` | signature | created | delegated | 0.000e+00 | - |
| SOFTMAX | float16 | `{}` | entry | upgraded | incorrect | 3.400e-01 | - |
| SOFTMAX | float32 | `{"dynamic_shape":false,"rank":2}` | signature | created | incorrect | 3.400e-01 | - |
| SQRT | float32 | `{}` | entry | conflict | delegated | 0.000e+00 | probe measured 'delegated' but the entry's 'fallback' comes from a full-model measurement (example-model-x); kept inferred — re-measure at full-model scale to resolve |
| TANH | float32 | `{}` | entry | unknown_result | unknown | - | - |

**Conflicts (probe vs full-model measurement; unresolved):**

- SQRT: probe measured 'delegated' but the entry's 'fallback' comes from a full-model measurement (example-model-x); kept inferred — re-measure at full-model scale to resolve

```
matrix diff — backend gpu_mldrift: 1.2.0 (2026-08-01) → 1.3.0 (2026-08-10)
  + RESHAPE [float32] {"dynamic_shape":false,"rank":2} (status=delegated)
  + SOFTMAX [float32] {"dynamic_shape":false,"rank":2} (status=incorrect)
  ~ SOFTMAX [float16] {}: status delegated → incorrect
  ~ SQRT [float32] {}: provenance measured → inferred
```

## Models re-linted (`cards_manifest_example.csv`)

### `model_mixed_example.tflite`

- **gpu_mldrift**: coverage 40.0% → 40.0%, partitions 1 → 1
  - s0/n3 SOFTMAX: delegated → incorrect

**Manifest errors (collected, not fail-fast):**

- `model_missing_example.tflite`: FileNotFoundError: No such file or directory
