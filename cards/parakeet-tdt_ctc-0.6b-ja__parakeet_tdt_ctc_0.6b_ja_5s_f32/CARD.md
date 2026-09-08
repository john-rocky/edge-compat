---
family: parakeet
license: cc-by-4.0
model_id: parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_f32
source_url: https://huggingface.co/litert-community/parakeet-tdt_ctc-0.6b-ja
task: automatic-speech-recognition
---

# parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_f32

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
| **Command** | `python stageA_ja.py && python stageB_ja.py --quant=none --output=parakeet_tdt_ctc_0.6b_ja_5s_f32.tflite  # two-process split of convert_to_tflite.py --model=nvidia/parakeet-tdt_ctc-0.6b-ja --quant=none --input_sec=5` |
| **Quantization** | none (float32 weights) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `parakeet_tdt_ctc_0.6b_ja_5s_f32.tflite` | `db103d352034fba5dfb43a8cbba9dba72afecc730ec5f9999200c6ea41a27a9b` | 2280.836 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-13/parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_f32.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-08-27/parakeet-tdt_ctc-0.6b-ja__parakeet_tdt_ctc_0.6b_ja_5s_f32__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 63.606 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 30.326 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, AOT host-compiled, SM8850 target) · Android 16 | 2026-08-27 | measured |

## Pitfalls

- Two signatures: encode (log-mel [1,80,500] -> [1,1024,63]) + decode (stateless 64-token TDT loop; logits [1,63,64,3078] = 3072 tokens + blank 3072 + 5 durations). 80 mel bins (v3 uses 128); NeMo preprocessing: preemph 0.97, n_fft 512, win 25 ms, hop 10 ms, per-feature norm (measured; HF card).
- Exact conversion: 28/28 five-second windows reproduce NeMo transcribe() (joined CER 0.0) on a 137 s CC0 test read; Pixel 8a GPU fully delegates both signatures (1862/1862 + 2083/2083 LITERT_CL) with identical greedy ids — encode 239 ms, decode 95 ms/call, ~20 calls per 5 s window (measured 2026-08-13).
- 2.4 GB float32 file — GPU compile on the Pixel 8a takes 25-37 s; the i8 sibling is 608 MB but fails Mali GPU compile (see the i8 meta) (measured).
- NeMo trap for anyone re-deriving references: model.transcribe() leaves decoder/joint in training mode (dropout on) — compute reference tensors before transcribe(), or re-.eval() after (measured; caused a false parity failure).
- Mali fp16 quirk: decode logits at token positions >= ~43 go NaN when the token suffix is zero-padded garbage (probe fixture artifact); real-prefix cells are clean and greedy decoding is unaffected (measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
