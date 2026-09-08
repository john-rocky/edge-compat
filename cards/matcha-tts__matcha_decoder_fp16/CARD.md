---
family: matcha-tts
license: mit
model_id: matcha-tts__matcha_decoder_fp16
source_url: https://huggingface.co/litert-community/Matcha-TTS
task: text-to-speech
---

# matcha-tts__matcha_decoder_fp16

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | matcha-tts |
| **Source** | https://huggingface.co/litert-community/Matcha-TTS |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python matcha/scripts/build_matcha.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `d2964252ab16d0494cf716a42ae9a1d74dfad6f492e6a19102ef2f76cfb15949.tflite` | `2c2e70235f0daaa9d82366416aad69b4f6e916f080a2da8b4b3b759f7cfd3069` | 21.567 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/matcha-tts__matcha_decoder_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 343.072 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 5965.4297349980925 | 9.553 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/matcha-tts__matcha_decoder_fp16__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 69.679 | - | - | - | 112.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- On the Pixel 8a the CFM decoder must run on CompiledModel CPU — a Mali ML Drift transformer-fusion bug (the text encoder + vocoder run on GPU); the Euler ODE loop runs host-side (zoo README L1005, L1010).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
