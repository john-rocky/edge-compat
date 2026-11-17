---
family: audio8-tts
license: apache-2.0
model_id: audio8-tts-preview-0.6b__codec_decoder_int8_t128
source_url: https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b
task: text-to-speech
---

# audio8-tts-preview-0.6b__codec_decoder_int8_t128

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | audio8-tts |
| **Source** | https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export-time PT2E dynamic int8 (export_codec_native_i8.py, the A/B partner of the post-hoc DRQ file under the house rule of 2026-07-22) + fix_native_i8_tables.py (fp32 codebooks written back into the flatbuffer); repro = hf-to-litertlm audio8_tts_work/ (REPRODUCE entry) litert-torch 0.9.4 (the interpreter named in every export/quantize script's usage line: torch 2.13.0, transformers 5.14.1, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0 — pip show 2026-09-28; REPRODUCE entry: 'env with litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer'); PyTorch oracle transformers 4.57.6 / torch 2.12.1 CPU fp32 |
| **Command** | `T=128 SUFFIX=_g2 python export_codec_native_i8.py && python fix_native_i8_tables.py out/codec/codec_decoder_i8native_T128_g2.tflite out/codec/codec_decoder_i8nativefix_T128_g2.tflite -> python assemble_ship.py out/ship (copied as codec_decoder_int8_T128.tflite; built_from codec/codec_decoder_i8nativefix_T128_g2.tflite) (REPRODUCE entry)` |
| **Quantization** | export-time (PT2E DYNAMIC) int8 for all convolutions and projections, fp32 RVQ codebook tables restored after export: the converter also quantized the unannotated codebook tables asymmetrically, which TFLite's EMBEDDING_LOOKUP kernel refuses at prepare (zero_point must be 0) and set_operator_type / module split do not prevent; corr 0.998 vs fp32, WER unchanged, ja speaker cosine 0.73 (fp16 0.74); 1.6x faster than fp16 on the Mac CPU (590 vs 967 ms at T128 in the scan) — 'the CPU option' (card File table + Accuracy) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `codec_decoder_int8_T128.tflite` | `8f7bb7d518bf77e33c3a729aeae55d96b79c592b465b60bd9f78f0143619d80b` | 126.295 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 90.0% |
| **Partitions** | 74 |
| **Blocking ops** | SIN, PAD, TRANSPOSE_CONV, TANH, LOGISTIC, SQUARED_DIFFERENCE, DEPTHWISE_CONV_2D, SLICE, RESHAPE, EMBEDDING_LOOKUP, MAXIMUM, MINIMUM |
| **Verdict provenance** | measured=892, unmatched=99 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-28/audio8-tts-preview-0.6b__codec_decoder_int8_t128.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-11-02/audio8-tts-preview-0.6b__codec_decoder_int8_t128__galaxy-s26.json`, `data/device_runs/2.3.2/2026-11-02/audio8-tts-preview-0.6b__codec_decoder_int8_t128__galaxy-s26.json`, `data/device_runs/2.3.2/2026-11-02/audio8-tts-preview-0.6b__codec_decoder_int8_t128__pixel-11-pro.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK 1 thread | 3978.71 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.2 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-11-02 | measured |
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK 1 thread; a short timing run started at the clean ready check (rested phone) | 3112.992 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-11-02 | measured |
| pixel-11-pro | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK 1 thread | 5731.868 | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert 2.3.2 · Android 17 (build CD1A.260618.001.A7) | 2026-11-02 | measured |

## Pitfalls

- CPU option only: on the Galaxy S26 GPU the int8 convolutions fail delegate kernel init ('Failed to create litert::ml_drift::DelegateKernelLiteRt: INVALID_ARGUMENT: Unable to parse bc coord for BATCH axis' — §4 prose; the line is not in the archived logs), so the GPU codec is the fp16 decoder (card Performance note).
- Post-hoc ai-edge-quantizer dynamic int8 on this decoder is not what ships: aeq's >32 MiB chunked path trips a numpy broadcast bug on the first decoder conv ([1536,1,7,1024], 44 MB); excluding that conv works (251 MB, corr 0.9984) but the native export is smaller (all convs int8) and faster at the same quality.
- Same windowing rule as the fp16 T128 decoder (one call = up to 128 frames = 5.9 s; windows are not sample-exact) (card Limitations).
- Sampled model: quantized graphs sample a different but valid trajectory, so per-frame agreement with the reference is not a meaningful metric; the gate is the transcript (whisper large-v3-turbo WER/CER vs the input text) and the voice (TitaNet-L speaker cosine vs the reference clip) on 14 seeded sentences (6 en + 6 ja with a cloned voice, one of each without). PyTorch reference en WER 1.1% / ja CER 0.0% / cosine 0.66 / 0.74; int8 slow + int8 fast + fp16 codec 1.1% / 0.0% / 0.68 / 0.76; int4 slow 1.1% / 0.0% / 0.63 / 0.77; int8 codec on the reference codes 1.1% / 0.0% / 0.67 / 0.73. The one English error is shared with the reference (the model drops the first word of one sentence). The fp32 graphs reproduce the reference code sequence frame for frame on all 14 sentences (card Accuracy).
- The graphs are driven by a host loop (audio8_tts_litert.py, Python, ai-edge-litert Interpreter, CPU): prompt construction, chunked prefill, the vendor's sampler (temperature 0.7 / top-p 0.9 / top-k 50, repetition-aware re-draw), 10 fast-AR calls per frame, windowed codec decode, voice registration; on a phone the same graphs run from Kotlin/C++ through the CompiledModel API — no on-device functional run exists yet, the Galaxy S26 rows are per-graph benchmark_model latency + delegate coverage, functional parity is Mac-only on the same files (card Files + Performance).
- Prompt length + generated frames must stay under 2,048 positions (the model's max_seq_len = the KV cache); the default cap is 512 frames (about 24 s) per call (card Limitations).
- GQA on the KV cache must not be expressed as repeat_interleave: it lowers to BROADCAST_TO (48 per step) and made decode 615 ms/step on the Mac (thread-count independent); folding the 7 query heads that share a kv head into the matmul row dimension removed it (16 ms/step fp32). RoPE: the vendor applies interleaved-pair rotation with a bf16-rounded table; the port permutes the q/k rows of wqkv to the rotate-half layout with the same bf16-rounded cos/sin as fp32 constants, bit-identical to the vendor buffers.
- fp16 weights give the AR graphs no speed or RAM gain on the CPU (XNNPACK unpacks them to fp32 at init), so the AR graphs ship int8/int4 and only the codec ships fp16.
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio (card Limitations).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
