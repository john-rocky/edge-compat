---
family: yolact
license: mit
model_id: yolact-resnet50
source_url: https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT
task: image-segmentation
---

# yolact-resnet50

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | yolact |
| **Source** | https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python yolact/scripts/build_yolact.py` |
| **Quantization** | none — fp32 (artifact tensors: 279 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `de20dcdb186be71368cbd668fd62e71e6089d0c94c100968975cfc6300c25a83.tflite` | `39aa8e5027e6638cfa3c9025854a9d3e5e95121a3bcebd926ab08973d32d0910` | 118.826 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/yolact-resnet50.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 1700.407 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.7315858145966206 | 63.108 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/yolact-resnet50__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/yolact-resnet50__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 38.367 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 14.344 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 1095.56 | - | - | - | 376.06 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Decode is host-side: SSD box decode against the 19248 priors shipped as priors.bin (variances [0.1,0.2]), per-class NMS, then lincomb masks sigmoid(proto @ coeff) cropped to each box (zoo README L676-678).
- Input is BGR with caffe-style normalization (x - [103.94,116.78,123.68]) / [57.38,57.12,58.40], no /255 (zoo README L678).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
