---
family: panns
license: cc-by-4.0
model_id: panns-cnn14-audioset
source_url: https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT
task: audio-classification
---

# panns-cnn14-audioset

| | |
|---|---|
| **Task** | audio-classification |
| **Family** | panns |
| **Source** | https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT |
| **License** | cc-by-4.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python panns/scripts/build_panns.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `91be10ebb0717b7c7793a3bcab086ea6b4dd95c047132dc31dfdf0362ddab1ab.tflite` | `26e52a285ff324778c7b90642e9733b47a46071622ecba93d42ca14ec19a467f` | 154.051 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/panns-cnn14-audioset.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2473.497 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 7.285172790579761e-06 | 13.235 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/panns-cnn14-audioset__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/panns-cnn14-audioset__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 12.154 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 3.888 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 390.821 | - | - | - | 632.66 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The log-mel front-end must run host-side: litert-torch lowers the 1024-tap DFT-conv wrongly (fp32 corr ~0.19) and |STFT|^2 overflows fp16 on Mali; the graph input is the precomputed log-mel [1,1,1001,64] (zoo README L1218).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
