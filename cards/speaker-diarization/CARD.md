---
family: wespeaker
license: mit
model_id: speaker-diarization
source_url: https://huggingface.co/litert-community/Speaker-Diarization-LiteRT
task: automatic-speech-recognition# HF card front matter pipeline_tag
---

# speaker-diarization

| | |
|---|---|
| **Task** | automatic-speech-recognition# HF card front matter pipeline_tag |
| **Family** | wespeaker |
| **Source** | https://huggingface.co/litert-community/Speaker-Diarization-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python diarization/scripts/build_diar.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `1c2f09bf17c4dc60dd830c02be355beb4a0042351bc6f6ebdb71957dde613e24.tflite` | `4853c284fdb3e39bd41f692d0bc3fd5068bf78782792d8ef02a3283c9d97554d` | 12.734 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/speaker-diarization.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 1399.142 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.00013505359290172167 | 3.852 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/speaker-diarization__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/speaker-diarization__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 6.05 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 1.993 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 186.062 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The kaldi-fbank front-end (hamming 25/10 ms, 80 mel, x2^15, CMN) runs host-side; the pyannote segmentation BiLSTM has no Mali GPU kernel and ships separately as ONNX for CPU (zoo README L1307-1313).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
