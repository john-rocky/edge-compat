---
family: matcha-tts
license: mit
model_id: matcha-tts__matcha_textenc_fp16
source_url: https://huggingface.co/litert-community/Matcha-TTS
task: text-to-speech
---

# matcha-tts__matcha_textenc_fp16

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
| `79369d1d9e007de126000f677ca13fadf0d93c8442c3608dcae0662be962a849.tflite` | `adf2642699b09a9f0cc302a64396554ac698401d93cffad4462792d6df8589a5` | 14.09 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/matcha-tts__matcha_textenc_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 229.993 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.009912348073582513 | 5.992 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/matcha-tts__matcha_textenc_fp16__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 38.679 | - | - | - | 112.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Fixed shapes (MAX_TEXT=256 phonemes) with a runtime float mask making padded positions a no-op, so one compiled graph handles any length (zoo README L1014).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
