---
family: dehazeformer
license: mit
model_id: dehazeformer-mct
source_url: https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT
task: image-to-image
---

# dehazeformer-mct

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | dehazeformer |
| **Source** | https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python dehaze/scripts/build_dehaze.py` |
| **Quantization** | none — fp32 (artifact tensors: 2535 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `6533ca6288d81f89e5cd96a0e96887324a539e28e1fcc7fb2f1739aea50a778b.tflite` | `ebd9756d55fbcf519a2887bfcef3b85788fec67fd81eb1642539fb8afa7c001e` | 16.0 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/dehazeformer-mct.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 709.245 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 1.348509457307736 | 96.34 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/dehazeformer-mct__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 2183.792 | - | - | - | 252.25 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The network outputs 72 per-pixel curve parameters at 256x256; the trilinear curve decode onto the full-resolution frame is a stated host-side step (zoo README L558, L564).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
