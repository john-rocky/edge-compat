---
family: rf-detr
license: apache-2.0
model_id: rf-detr-seg-nano__rfdetrseg_graphb_fp16
source_url: https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT
task: image-segmentation
---

# rf-detr-seg-nano__rfdetrseg_graphb_fp16

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | rf-detr |
| **Source** | https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python scripts/build_rfdetrseg_split.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `2609a49a19d7f7d86177f6282cc80e3b445204ad2b1064064b361ec8b01f1255.tflite` | `71ad26d20f3f5bc15b8fc73384a1ce3ad0ceb6b4f994bb042662641aea071ae7` | 14.099 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/rf-detr-seg-nano__rfdetrseg_graphb_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 445.383 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 15.320825182287036 | 16.68 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rf-detr-seg-nano__rfdetrseg_graphb_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/rf-detr-seg-nano__rfdetrseg_graphb_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 22.663 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 64.07 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 324.54 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- query_feat is a HOST-FED input (query_feat.bin published beside the model), not a baked constant — the delegate's baked-constant execution bug — HF card.
- masks are per-query FULL-IMAGE 78x78 raw logits (inside = logit > 0), upsampled bilinearly on the host; logits are 91-way COCO id space (id 0 unused) — HF card 'Preprocessing / outputs'.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
