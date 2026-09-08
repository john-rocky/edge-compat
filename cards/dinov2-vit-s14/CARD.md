---
family: dinov2
license: apache-2.0
model_id: dinov2-vit-s14
source_url: https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT
task: image-feature-extraction
---

# dinov2-vit-s14

| | |
|---|---|
| **Task** | image-feature-extraction |
| **Family** | dinov2 |
| **Source** | https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python dinov2/scripts/build_dinov2.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `854a975b2cba85ab2e276ecb59cdb3c13f5f6058206c4612b218cc6bc7a1476f.tflite` | `c0697093316b777de2c79ed895f74c864136840309ecb2aab6f51cbe0c228e7f` | 42.814 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/dinov2-vit-s14.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2977.405 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 2.1121107354221573 | 29.86 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
