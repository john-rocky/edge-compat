---
family: wav2vec2
license: apache-2.0
model_id: wav2vec2-base-960h__w2v2_asr_head_fp16
source_url: https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT
task: automatic-speech-recognition
---

# wav2vec2-base-960h__w2v2_asr_head_fp16

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | wav2vec2 |
| **Source** | https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python build_w2v2_asr.py all` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `9e4b518e0dd9c9a54b1098d367ce85a85a17c6e1b29e43e6f2f8e9496f8f034c.tflite` | `ed4ef4d37dc12a58e40e8d964d7cfac5d880def72b9431d504c4c3671d575ad9` | 171.86 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/wav2vec2-base-960h__w2v2_asr_head_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | pass | yes | - | - | 51.323 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/wav2vec2-base-960h__w2v2_asr_head_fp16__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 2820.028 | - | - | - | 640.2 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Two-graph split (frontend conv feature extractor + transformer/CTC head); features hand off as [1,799,768] between the graphs — HF card I/O table.
- Fixed 16 s waveform window [1,256000]; greedy CTC decode on the host — HF card usage snippet.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
