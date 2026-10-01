---
family: audio8-tts
license: apache-2.0
model_id: audio8-tts-preview-0.6b__slow_ar_int4
source_url: https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b
task: text-to-speech
---

# audio8-tts-preview-0.6b__slow_ar_int4

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | audio8-tts |
| **Source** | https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (fp32 export of the torch port arktts_port.py) + ai-edge-quantizer post-hoc blockwise-32 OCTAV int4 (quantize_ar.py bo4); repro = hf-to-litertlm audio8_tts_work/ (REPRODUCE entry) litert-torch 0.9.4 (~/venvs/lt094dev, the interpreter named in every export/quantize script's usage line: torch 2.13.0, transformers 5.14.1, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0 — pip show 2026-09-28; REPRODUCE entry: 'env with litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer'); PyTorch oracle transformers 4.57.6 / torch 2.12.1 CPU fp32 in ~/parakeet-env (FINDINGS §1) |
| **Command** | `PREFILL=256 SUFFIX=_p256 python export_slow.py -> python quantize_ar.py out/slow/slow_fp32_c2048_p256.tflite bo4 -> python assemble_ship.py out/ship (copied as slow_ar_int4.tflite; built_from slow/slow_bo4_c2048_p256.tflite) (REPRODUCE entry)` |
| **Quantization** | blockwise-32 OCTAV int4 FULLY_CONNECTED + int8 EMBEDDING_LOOKUP tables (quantize_ar.py bo4; card File table 'blockwise-32 OCTAV int4 projections'); teacher-forced vs oracle: slow logits max|d| 5.5-7.7, argmax 66-76% (FINDINGS §4); end to end it keeps the oracle's WER/CER (en 1.1%, ja 0.0%) with en speaker cosine 0.63 (oracle 0.66) and ja 0.77 (card Accuracy; NOTES gate table) — 'the small option' |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `slow_ar_int4.tflite` | `bd2ff1a9b1ec98c064e0e801162e1ae8cd83f928574df527a37be23175625458` | 368.475 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 93.5% |
| **Partitions** | 104 |
| **Blocking ops** | LOGISTIC, DYNAMIC_UPDATE_SLICE, PACK, GREATER_EQUAL, LESS_EQUAL, LOGICAL_AND, RESHAPE, SELECT_V2, SLICE, MAXIMUM, MINIMUM, ADD, EMBEDDING_LOOKUP, LESS, SELECT, GATHER_ND |
| **Verdict provenance** | measured=2883, unmatched=199 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-28/audio8-tts-preview-0.6b__slow_ar_int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:pipeline slow_ar_int4 + fast_ar_int8 + codec_decoder_fp16_T128 (host loop, CPU 4 threads) | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-09-28 | measured |

## Pitfalls

- Same graph contract as slow_ar_int8 (signatures prefill_256 + decode, KV cache 2048 as I/O, 4,097 logits + hidden state); select it with `--slow slow_ar_int4.tflite` in the host loop (card Quick start).
- int4 prefill is slower than int8 on the S26 CPU (prefill_256 417 ms vs 241 ms) while decode is faster (10.0 vs 12.3 ms); Mac CPU decode 12.2 vs 9.0 ms per frame (card Performance; NOTES S26 clean legs).
- Sampled model: quantized graphs sample a different but valid trajectory, so per-frame agreement with the reference is not a meaningful metric; the gate is the transcript (whisper large-v3-turbo WER/CER vs the input text) and the voice (TitaNet-L speaker cosine vs the reference clip) on 14 seeded sentences (6 en + 6 ja with a cloned voice, one of each without). PyTorch reference en WER 1.1% / ja CER 0.0% / cosine 0.66 / 0.74; int8 slow + int8 fast + fp16 codec 1.1% / 0.0% / 0.68 / 0.76; int4 slow 1.1% / 0.0% / 0.63 / 0.77; int8 codec on the reference codes 1.1% / 0.0% / 0.67 / 0.73. The one English error is shared with the reference (the model drops the first word of one sentence). The fp32 graphs reproduce the reference code sequence frame for frame on all 14 sentences (card Accuracy; FINDINGS §2, §5).
- The graphs are driven by a host loop (audio8_tts_litert.py, Python, ai-edge-litert Interpreter, CPU): prompt construction, chunked prefill, the vendor's sampler (temperature 0.7 / top-p 0.9 / top-k 50, repetition-aware re-draw), 10 fast-AR calls per frame, windowed codec decode, voice registration; on a phone the same graphs run from Kotlin/C++ through the CompiledModel API — no on-device functional run exists yet, the Galaxy S26 rows are per-graph benchmark_model latency + delegate coverage, functional parity is Mac-only on the same files (card Files + Performance; FINDINGS §8).
- Prompt length + generated frames must stay under 2,048 positions (the model's max_seq_len = the KV cache); the default cap is 512 frames (about 24 s) per call (card Limitations).
- GQA on the KV cache must not be expressed as repeat_interleave: it lowers to BROADCAST_TO (48 per step) and made decode 615 ms/step on the Mac (thread-count independent); folding the 7 query heads that share a kv head into the matmul row dimension removed it (16 ms/step fp32). RoPE: the vendor applies interleaved-pair rotation with a bf16-rounded table; the port permutes the q/k rows of wqkv to the rotate-half layout with the same bf16-rounded cos/sin as fp32 constants, bit-identical to the vendor buffers (FINDINGS §2).
- fp16 weights give the AR graphs no speed or RAM gain on the CPU (XNNPACK unpacks them to fp32 at init), so the AR graphs ship int8/int4 and only the codec ships fp16 (FINDINGS §4).
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio (card Limitations).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
