---
family: audio8-tts
license: apache-2.0
model_id: audio8-tts-preview-0.6b__codec_encoder_fp16_10s
source_url: https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b
task: text-to-speech
---

# audio8-tts-preview-0.6b__codec_encoder_fp16_10s

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | audio8-tts |
| **Source** | https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (fp32 export of the patched vendor codec encoder, static 10 s bucket) + ai-edge-quantizer fp16 FLOAT_CASTING (quantize_encoder.py fp16); repro = hf-to-litertlm audio8_tts_work/ (REPRODUCE entry) litert-torch 0.9.4 (~/venvs/lt094dev, the interpreter named in every export/quantize script's usage line: torch 2.13.0, transformers 5.14.1, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0 — pip show 2026-09-28; REPRODUCE entry: 'env with litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer'); PyTorch oracle transformers 4.57.6 / torch 2.12.1 CPU fp32 in ~/parakeet-env (FINDINGS §1) |
| **Command** | `SEC=10 SUFFIX=_g2 python export_codec_encoder.py (audio [1,1,442368] fp32 -> codes [1,10,216] int32; N = ceil(10*44100/2048)*2048; verified vs the oracle reference codes) -> python quantize_encoder.py out/codec/codec_encoder_fp32_10s_g2.tflite fp16 -> python assemble_ship.py out/ship (copied as codec_encoder_fp16_10s.tflite; built_from codec/codec_encoder_fp16_10s_g2.tflite) (REPRODUCE entry; export_codec_encoder.py docstring)` |
| **Quantization** | fp16 weights (ai-edge-quantizer FLOAT_CASTING), fp32 compute; codec encoder codes 100% equal to the oracle at fp32 (FINDINGS §2); the bundled voices' codes were produced by the PyTorch fp32 encoder and agree with this graph on 99.4-99.8% of entries (voices/*/meta.json; NOTES ship smoke: register-voice = 99.8% codes vs oracle) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `codec_encoder_fp16_10s.tflite` | `e27deb6663e25919d2149c63ff70e10418f0f341df963d2ae458d1466359a977` | 399.846 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 83.5% |
| **Partitions** | 210 |
| **Blocking ops** | DEQUANTIZE, SIN, PAD, DEPTHWISE_CONV_2D, LOGISTIC, SQUARED_DIFFERENCE, RESHAPE, ARG_MAX, CAST, EMBEDDING_LOOKUP, CONCATENATION, SQRT |
| **Verdict provenance** | measured=1470, unmatched=128 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-28/audio8-tts-preview-0.6b__codec_encoder_fp16_10s.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |

## Pitfalls

- Voice registration accepts up to 10 s of 44.1 kHz reference audio (one static bucket, 442,368 samples = 10.03 s -> 10 x 216 codes); the reference transcript must match the audio (card Limitations + Quick start `--register-voice`).
- CPU graph on Android: the RVQ argmax/cast run on int64 tensors, which the S26 OpenCL delegate refuses ('ARG_MAX: Tensor type(INT64) is not supported', 'CAST: Tensor type(INT64) is not supported') (NOTES S26 clean legs); the card's S26 row is CPU 4 threads.
- Sampled model: quantized graphs sample a different but valid trajectory, so per-frame agreement with the reference is not a meaningful metric; the gate is the transcript (whisper large-v3-turbo WER/CER vs the input text) and the voice (TitaNet-L speaker cosine vs the reference clip) on 14 seeded sentences (6 en + 6 ja with a cloned voice, one of each without). PyTorch reference en WER 1.1% / ja CER 0.0% / cosine 0.66 / 0.74; int8 slow + int8 fast + fp16 codec 1.1% / 0.0% / 0.68 / 0.76; int4 slow 1.1% / 0.0% / 0.63 / 0.77; int8 codec on the reference codes 1.1% / 0.0% / 0.67 / 0.73. The one English error is shared with the reference (the model drops the first word of one sentence). The fp32 graphs reproduce the reference code sequence frame for frame on all 14 sentences (card Accuracy; FINDINGS §2, §5).
- The graphs are driven by a host loop (audio8_tts_litert.py, Python, ai-edge-litert Interpreter, CPU): prompt construction, chunked prefill, the vendor's sampler (temperature 0.7 / top-p 0.9 / top-k 50, repetition-aware re-draw), 10 fast-AR calls per frame, windowed codec decode, voice registration; on a phone the same graphs run from Kotlin/C++ through the CompiledModel API — no on-device functional run exists yet, the Galaxy S26 rows are per-graph benchmark_model latency + delegate coverage, functional parity is Mac-only on the same files (card Files + Performance; FINDINGS §8).
- Prompt length + generated frames must stay under 2,048 positions (the model's max_seq_len = the KV cache); the default cap is 512 frames (about 24 s) per call (card Limitations).
- GQA on the KV cache must not be expressed as repeat_interleave: it lowers to BROADCAST_TO (48 per step) and made decode 615 ms/step on the Mac (thread-count independent); folding the 7 query heads that share a kv head into the matmul row dimension removed it (16 ms/step fp32). RoPE: the vendor applies interleaved-pair rotation with a bf16-rounded table; the port permutes the q/k rows of wqkv to the rotate-half layout with the same bf16-rounded cos/sin as fp32 constants, bit-identical to the vendor buffers (FINDINGS §2).
- fp16 weights give the AR graphs no speed or RAM gain on the CPU (XNNPACK unpacks them to fp32 at init), so the AR graphs ship int8/int4 and only the codec ships fp16 (FINDINGS §4).
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio (card Limitations).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
