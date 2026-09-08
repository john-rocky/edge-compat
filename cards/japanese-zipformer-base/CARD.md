---
family: zipformer
license: apache-2.0
model_id: japanese-zipformer-base
source_url: https://huggingface.co/litert-community/japanese-zipformer-base-LiteRT
task: automatic-speech-recognition
---

# japanese-zipformer-base

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | zipformer |
| **Source** | https://huggingface.co/litert-community/japanese-zipformer-base-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python build_ja_zipformer.py all` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `a63277edf68e02c13e22938c5eb3a2ec27d663eb55df9758ee896494a54ba414.tflite` | `c77532b76f5af17421bf1aaecb0ff8acb328e9342ed07960c99badb365ce1dca` | 188.154 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/japanese-zipformer-base.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | pass | yes | - | - | 61.167 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/japanese-zipformer-base__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 2641.607 | - | - | - | 926.14 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is the raw 16 kHz waveform [1,256000] (16 s window) plus 4 additive mask-bias inputs; output is CTC logits [1,799,3004] (BPE-3004 vocab) — HF card I/O table.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
