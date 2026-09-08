---
family: gfpgan
license: apache-2.0
model_id: gfpgan-v1.4__gfpgan_fp16
source_url: https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT
task: image-to-image
---

# gfpgan-v1.4__gfpgan_fp16

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | gfpgan |
| **Source** | https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python build_gfpgan.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `7348d3e5f97cf3f0a855a7ef13a3d0b0f091c2891930390beef85c1068696f07.tflite` | `6b792e1c33eb93b8115c57086042a440adf4072c60349d85beb266821d2b3dfe` | 411.504 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/gfpgan-v1.4__gfpgan_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 10.412 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 903550624.8474121 | 60.61 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/gfpgan-v1.4__gfpgan_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/gfpgan-v1.4__gfpgan_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 88.949 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 32.258 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 4085.305 | - | - | - | 1309.08 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input must be an FFHQ-aligned 512x512 face crop (YuNet 5-landmark similarity warp to the facexlib template); the StyleGAN prior mangles the mouth on off-template crops (zoo README L1592, L1600; HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
