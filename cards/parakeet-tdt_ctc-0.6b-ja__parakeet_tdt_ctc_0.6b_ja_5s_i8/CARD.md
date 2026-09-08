---
family: parakeet
license: cc-by-4.0
model_id: parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_i8
source_url: https://huggingface.co/litert-community/parakeet-tdt_ctc-0.6b-ja
task: automatic-speech-recognition
---

# parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_i8

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | parakeet |
| **Source** | https://huggingface.co/litert-community/parakeet-tdt_ctc-0.6b-ja |
| **License** | cc-by-4.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via the official litert-samples speech_recognition convert pipeline (ParakeetTDT path; ParakeetTDTCTCJa subclass, litert-samples#277) 0.10.0 (editable dev checkout 115a136) |
| **Command** | `python stageA_ja.py && python stageB_ja.py --quant=drq --output=parakeet_tdt_ctc_0.6b_ja_5s_i8.tflite  # two-process split of convert_to_tflite.py --model=nvidia/parakeet-tdt_ctc-0.6b-ja --quant=drq --input_sec=5` |
| **Quantization** | int8 dynamic-range channelwise weights, fp32 activations (the pipeline's drq recipe) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `parakeet_tdt_ctc_0.6b_ja_5s_i8.tflite` | `6ab11f31e97f21b8bc5454be6ed59c66413ce7329ba14f9fd2930398587229c8` | 579.747 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-13/parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_i8.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 1184.507 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_i8__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-27/parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_i8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | load_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | npu_qnn | load_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, AOT host-compiled, SM8850 target) · Android 16 | 2026-08-27 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | sig:decode | 110.356 | - | - | - | 1539.55 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | sig:encode | 275.894 | - | - | - | 1422.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Two signatures: encode (log-mel [1,80,500] -> [1,1024,63]) + decode (stateless 64-token TDT loop; logits [1,63,64,3078] = 3072 tokens + blank 3072 + 5 durations). 80 mel bins (v3 uses 128); NeMo preprocessing: preemph 0.97, n_fft 512, win 25 ms, hop 10 ms, per-feature norm (measured; HF card).
- This i8 variant FAILS GPU compile on Mali (Pixel 8a): ML Drift 'Unable to parse bc coord for BATCH axis', litert 2.1.3 and 2.1.5; the f32 sibling compiles and runs on the same GPU. On Mali devices run i8 on CPU (verified: encode 1157 ms, decode 380 ms/call) (measured 2026-08-13).
- i8 fidelity vs the fp32/NeMo output: 14/28 five-second windows identical, joined CER 6.1% on a 137 s CC0 test read; the f32 variant is exact (28/28, CER 0.0) (measured).
- NeMo trap for anyone re-deriving references: model.transcribe() leaves decoder/joint in training mode (dropout on) — compute reference tensors before transcribe(), or re-.eval() after (measured; caused a false parity failure).
- tokenizer.json in the repo is converted from the NeMo SentencePiece model (nvidia repo ships only the .nemo); decode parity vs SentencePiece verified on 200 random id sequences + real text (measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
