---
family: parakeet
license: cc-by-4.0
model_id: parakeet-tdt-0.6b-v3__parakeet_tdt_0.6b_v3_30s_i8_stateful
source_url: https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-v3-LiteRT
task: automatic-speech-recognition
---

# parakeet-tdt-0.6b-v3__parakeet_tdt_0.6b_v3_30s_i8_stateful

| | |
|---|---|
| **Task** | automatic-speech-recognition |
| **Family** | parakeet |
| **Source** | https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-v3-LiteRT |
| **License** | cc-by-4.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via the official litert-samples speech_recognition convert pipeline (compiled_model_api/speech_recognition/convert/convert_to_tflite.py at efca5805, 2026-05-15) litert-torch 0.9.4, NeMo 3.0.0 |
| **Command** | `python compiled_model_api/speech_recognition/convert/convert_to_tflite.py --model nvidia/parakeet-tdt-0.6b-v3 --input_sec 30 --stateful_after 4 --quant drq --sample_audio <wav> --output parakeet_tdt_0.6b_v3_30s_i8_stateful.tflite` |
| **Quantization** | int8 dynamic-range weights, fp32 activations (the pipeline's drq recipe); the same recipe as litert-community/parakeet-tdt-0.6b-v3's 5 s file, with a 30 s window |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `parakeet_tdt_0.6b_v3_30s_i8_stateful.tflite` | `378935f8897f0c713ad2aa97d939c73c44e9f26546e12c1cd6af9c76f386f06a` | 600.519 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-25/parakeet-tdt-0.6b-v3__parakeet_tdt_0.6b_v3_30s_i8_stateful.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 7451.368 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-25 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/selfbuilt-66058c82+omni-eval-patches/2026-09-25/parakeet-tdt-0.6b-v3__parakeet_tdt_0.6b_v3_30s_i8_stateful__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | 1825.7 | Mac Studio (Apple M4 Max, 128 GB) · Apple M4 Max · litert-lm selfbuilt-66058c82+omni-eval-patches · macOS 27.0 (26A428) | 2026-09-25 | measured |

## Pitfalls

- Three signatures: encode (log-mel [1,128,3000] -> [1,1024,375]) + decode (4-token stateless prefix [1,4], LSTM state 2 x [2,1,640]; logits [1,375,4,8198]) + decode_1 (single step [1,1]; logits [1,375,1,8198]); 123 subgraphs (measured with the repo parser and ai-edge-litert 2.2.0).
- Why 30 s: through LiteRT-LM omni/asr every utterance is cut into standalone windows of the export's length, and the original model returns no text for 37 % of standalone 5 s windows that start mid-utterance (267 of 724, NeMo fp32, LibriSpeech test-clean), so the 5 s file scores 17.67 / 18.25 WER (test-clean / test-other) against NeMo's 1.92 / 3.59. With a 30 s window a LibriSpeech utterance is one window (measured 2026-09-25).
- Through LiteRT-LM omni/asr at main@66058c82 this file needs three patches (edge-llm-bench tools/omni-eval/patches): 01 Slaney/power mel, 03 a parakeet-tdt-0.6b-v3-30s metadata entry (inputMilliseconds 30000, nFrames 3000, this file's URL), 06 TdtDecoder stops at the last encoder frame that carries audio, with a 10-symbols-per-step cap. Without 06 a short clip is decoded over up to 28 s of zero padding and the TDT decoder adds text there: 2.69 / 10.54 WER; with 06: 2.17 / 5.35 (Mac CPU, 4 threads, Open ASR Leaderboard scoring; measured).
- Cost of the long window: every utterance pays a 30 s encoder pass, RTFx 17-19 on an M4 Max CPU against 34 for the 5 s file. What remains at 2.17 / 5.35 is clips under 5 s on test-other (10.2 % WER, 44 of 1419 empty), the model's own behaviour on short standalone input (measured).
- Mel front-end must match NeMo: Slaney scale, power spectrum, floor 5.96e-8, per-feature normalization over the whole padded window. The export was trained on padded batches without a length mask, so normalizing over the valid frames only and zero-filling the rest sends short clips from 2.40 to 11.42 WER on the 5 s file (measured).
- Engine note, not the model: through omni/asr on Metal (Mac GPU) RSS grows about 10 MB per session on the 5 s file (decoder output buffers not released); reuse one session for a batch. The 30 s file was measured on CPU only.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
