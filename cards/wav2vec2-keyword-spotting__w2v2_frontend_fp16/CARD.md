---
family: wav2vec2
license: apache-2.0
model_id: wav2vec2-keyword-spotting__w2v2_frontend_fp16
source_url: https://huggingface.co/litert-community/wav2vec2-keyword-spotting
task: audio-classification
---

# wav2vec2-keyword-spotting__w2v2_frontend_fp16

| | |
|---|---|
| **Task** | audio-classification |
| **Family** | wav2vec2 |
| **Source** | https://huggingface.co/litert-community/wav2vec2-keyword-spotting |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python wav2vec2-kws/scripts/build_w2v2_split.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `6df59f7a42d94008dfb39940e27fa82425c8e9a7b8a3ac6b2c8ef1e7d705ff68.tflite` | `3ea55a43a61b8ca57b2663a9e0d33f92cde669fe0bbd1fcbf5daa1570a6cd54e` | 8.818 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/wav2vec2-keyword-spotting__w2v2_frontend_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 309.91 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.07765237020316026 | 3.365 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/wav2vec2-keyword-spotting__w2v2_frontend_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/wav2vec2-keyword-spotting__w2v2_frontend_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 5.285 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | run_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 106.245 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The model ships as two graphs (frontend + head) because the full 1008-node graph exceeds the Mali shader-compile limit; both halves are needed for a prediction (zoo README L1195).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
