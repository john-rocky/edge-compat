---
family: pp-ocrv5
license: apache-2.0
model_id: pp-ocrv5__ppocr_rec_fp16
source_url: https://huggingface.co/litert-community/PP-OCRv5-LiteRT
task: text-recognition
---

# pp-ocrv5__ppocr_rec_fp16

| | |
|---|---|
| **Task** | text-recognition |
| **Family** | pp-ocrv5 |
| **Source** | https://huggingface.co/litert-community/PP-OCRv5-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (weights via the PaddleOCR2Pytorch port) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python ppocr/scripts/build_rec.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `de2edc3b5726675344b8e701957f2b5e803e7374c6c693347ea4817812077b8a.tflite` | `ef7bb5aba20a1717101f0f112dd8cb1ed8b043ebaf96af05a395c6905ca66456` | 16.378 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/pp-ocrv5__ppocr_rec_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 449.3 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 3.5547966028266442 | 15.125 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/pp-ocrv5__ppocr_rec_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/pp-ocrv5__ppocr_rec_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 7.998 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 5.911 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 86.447 | - | - | - | 135.48 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- CTC decode runs on CPU after the recognizer (zoo README L1566).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
