---
family: dewarpnet
license: mit
model_id: dewarpnet
source_url: https://huggingface.co/litert-community/DewarpNet-LiteRT
task: image-to-image
---

# dewarpnet

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | dewarpnet |
| **Source** | https://huggingface.co/litert-community/DewarpNet-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python dewarp/scripts/build_dewarp.py` |
| **Quantization** | none — fp32 (artifact tensors: 541 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `9c3cea33c74829b37dafb94bf625fd11f18a3592e1a65d9171576acfcec952a2.tflite` | `0f72fc016f50bd16cc2055e0b499a4317702ed81b3ff6672fd9affe0bad6d064` | 180.104 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/dewarpnet.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 633.472 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.015679701433383975 | 22.56 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/dewarpnet__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 593.957 | - | - | - | 490.2 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is BGR, x/255, NCHW; the grid_sample unwarp using the predicted backward map is a stated host-side step (zoo README L502-504).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
