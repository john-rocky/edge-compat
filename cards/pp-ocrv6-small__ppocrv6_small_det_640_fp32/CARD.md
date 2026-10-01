---
family: pp-ocrv6-small
license: apache-2.0
model_id: pp-ocrv6-small__ppocrv6_small_det_640_fp32
source_url: https://huggingface.co/litert-community/PP-OCRv6-Small-LiteRT
task: image-to-text
---

# pp-ocrv6-small__ppocrv6_small_det_640_fp32

| | |
|---|---|
| **Task** | image-to-text |
| **Family** | pp-ocrv6-small |
| **Source** | https://huggingface.co/litert-community/PP-OCRv6-Small-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 (litert-converter 0.4.0; torch 2.13.0; Python 3.12.12 per conversion/README.md) |
| **Command** | `.venv/bin/python conversion/build.py det` |
| **Quantization** | none — fp32 (conversion/README.md: 'Reproduce the FP32 conversion'; the card: 'FP16 and INT8 artifact variants were not evaluated') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `ppocrv6_small_det_640_fp32.tflite` | `ec9430b6e592ffb28e83b6bab539262a30d267d2475c136b193926ad83835768` | 11.807 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 4.5% |
| **Partitions** | 13 |
| **Blocking ops** | CONV_2D, ADD, DEPTHWISE_CONV_2D, PAD, SUM, MUL, RESHAPE, SUB, TRANSPOSE, MAX_POOL_2D, CONCATENATION, RELU, RESIZE_NEAREST_NEIGHBOR, SLICE, LOGISTIC |
| **Verdict provenance** | measured=13, unmatched=273 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-25/pp-ocrv6-small__ppocrv6_small_det_640_fp32.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 214.418 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |
| webgpu_mldrift | pass | yes | pass | 4.0046921755878557e-05 | 10.045 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |

## Pitfalls

- FP32 model storage and GPU computation precision are separate settings: the same four files passed only 27/37 tensor cases at the runtime's default GPU precision and 37/37 with explicit FP32 (card, 'FP32 model storage and GPU computation precision are separate settings').
- NPU execution completed but its parity failed (19/37 tensor cases, 0/6 complete image cases); NPU is not a supported configuration for this release (card).
- Outputs are probabilities; do not apply another softmax. The 18,710-entry dictionary already contains blank at index 0 and space at the final index; greedy CTC collapses repeats then drops blank (card, recognizer usage).
- Graph rewrites are exact re-implementations, not approximations: the detector's two k2/s2 transposed convolutions became zero stuffing + flipped convolution, fused QKV split into 4D branches, spatial means split, clamps / hard sigmoid as ReLU differences; exact GELU retained (conversion/README.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
