---
family: zipformer
license: apache-2.0
model_id: zipformer-medium-cr-ctc__zipformer_ctc_fp16
source_url: https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT
task: automatic-speech-recognition
---

# zipformer-medium-cr-ctc__zipformer_ctc_fp16

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
| `b3b4e3b342c9a08e25e8496a9b473ab8e028c2216e2a12d3f95793ccd18c4bab.tflite` | `9515d94c04306798fecfe695eb8f79cc8ac98ed3d8eab81c28ac464b7dbddf48` | 125.4 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/zipformer-medium-cr-ctc__zipformer_ctc_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2190.745 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 81159.26119402985 | 25.19 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/zipformer-medium-cr-ctc__zipformer_ctc_fp16__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 762.79 | - | - | - | 529.12 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is host-computed kaldi-fbank [1,1600,80] with the waveform kept in [-1,1] scale, plus 4 additive attention-bias inputs (0 = valid, -1000 = pad; one per downsampling rate) — HF card 'How it runs'.
- Output is raw CTC logits at 25 Hz (blank id 0): LogSoftmax is deliberately NOT in the graph (its REDUCE_MAX lowering fails the on-device GPU compile); greedy CTC is argmax-invariant — HF card.
- Fixed 16 s window; shorter audio is padded with log(1e-10) fbank frames — HF card.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
