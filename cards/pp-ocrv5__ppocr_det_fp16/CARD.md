---
family: pp-ocrv5
license: apache-2.0
model_id: pp-ocrv5__ppocr_det_fp16
source_url: https://huggingface.co/litert-community/PP-OCRv5-LiteRT
task: text-detection
---

# pp-ocrv5__ppocr_det_fp16

| | |
|---|---|
| **Task** | text-detection |
| **Family** | pp-ocrv5 |
| **Source** | https://huggingface.co/litert-community/PP-OCRv5-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (weights via the PaddleOCR2Pytorch port) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python ppocr/scripts/build_det.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `d4e2f90e59e429080970fdc515e2910f69888a6000a6f35e6c18cfa3677cb24f.tflite` | `b635b1d7f0e171a19beda7e8f386a62d4f0a3a1c46ed42a09603a902f2059ccc` | 9.627 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/pp-ocrv5__ppocr_det_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 916.36 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 2.253396026253472e-05 | 12.325 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/pp-ocrv5__ppocr_det_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/pp-ocrv5__ppocr_det_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 17.59 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 8.634 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 460.168 | - | - | - | 205.59 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- DB box postprocess runs on CPU between detection and recognition (zoo README L1566).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
