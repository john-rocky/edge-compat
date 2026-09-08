---
family: zipformer
license: apache-2.0
model_id: zipformer-medium-cr-ctc__zipformer_ctc_small_fp16
source_url: https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT
task: automatic-speech-recognition
---

# zipformer-medium-cr-ctc__zipformer_ctc_small_fp16

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | zipformer |
| **Source** | https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python build_zipformer_ctc.py all` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `1c45985b9f7293b2cd52df0ede49e2bb9c3e2d760a6a730fba282efd0f3c8e2c.tflite` | `ff6d70c7e8cfdfcf994be625456ca6543e0dbcc76fd4fd428e2138642d0852ba` | 44.076 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/zipformer-medium-cr-ctc__zipformer_ctc_small_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 1433.62 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 11323.859623733719 | 14.132 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/zipformer-medium-cr-ctc__zipformer_ctc_small_fp16__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 606.175 | - | - | - | 276.06 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is host-computed kaldi-fbank [1,1600,80] with the waveform kept in [-1,1] scale, plus 4 additive attention-bias inputs (0 = valid, -1000 = pad; one per downsampling rate) — HF card 'How it runs'.
- Output is raw CTC logits at 25 Hz (blank id 0): LogSoftmax is deliberately NOT in the graph (its REDUCE_MAX lowering fails the on-device GPU compile); greedy CTC is argmax-invariant — HF card.
- Fixed 16 s window; shorter audio is padded with log(1e-10) fbank frames — HF card.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
