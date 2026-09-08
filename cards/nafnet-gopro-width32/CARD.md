---
family: nafnet
license: mit
model_id: nafnet-gopro-width32
source_url: https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT
task: image-to-image
---

# nafnet-gopro-width32

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | nafnet |
| **Source** | https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python nafnet/scripts/build_nafnet.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `c267178696fbc39bb1584b1694e755527a8af5e16db9a74b4b8978a7a72dfd9f.tflite` | `8e749782f7c8a6e462ba7b457724edd089216fa3a25f9d5f83d0caaae7158507` | 36.539 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/nafnet-gopro-width32.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2712.535 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.013754045307443365 | 20.842 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/nafnet-gopro-width32__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/nafnet-gopro-width32__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 30.894 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 18.583 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 820.934 | - | - | - | 264.3 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
