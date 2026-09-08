---
family: tipsv2
license: apache-2.0
model_id: tipsv2-b14-dpt
source_url: https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT
task: depth-estimation
---

# tipsv2-b14-dpt

| | |
|---|---|
| **Task** | depth-estimation |
| **Family** | tipsv2 |
| **Source** | https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python build_tipsv2.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `bd722afe6b9cc13b0de41c2717df9c1decda6fa5a9fb98e9513a2dda1761ba8e.tflite` | `2ba75bcd917aed7eae5321d02d827036f0d40eb3f604e900f86015bcd70f5f4a` | 303.591 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/tipsv2-b14-dpt.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 148.665 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 23054670333.862305 | 285.11 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Pitfalls

- One graph, three heads: depth + surface normals + ADE20K segmentation from a single TIPSv2-B/14 backbone [1,3,448,448] — HF card.
- Depth-head fp16 range fold: the depth decoder's activations reach ~1e8 at the logits (fp16 max 65504); the decoder is a ReLU/affine chain ending in a scale-invariant normalisation, so a power-of-2 rescale is folded through it (exact) — HF card 'Conversion' notes.
- SafeLayerNorm scales by 1/64 before squaring so the fp16 variance cannot overflow; tanh-GELU replaces ERF — HF card.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
