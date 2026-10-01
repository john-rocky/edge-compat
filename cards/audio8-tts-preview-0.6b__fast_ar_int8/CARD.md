---
family: audio8-tts
license: apache-2.0
model_id: audio8-tts-preview-0.6b__fast_ar_int8
source_url: https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b
task: text-to-speech
---

# audio8-tts-preview-0.6b__fast_ar_int8

| | |
|---|---|
| **Task** | text-to-speech |
| **Family** | audio8-tts |
| **Source** | https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (fp32 export of the torch port arktts_port.py) + ai-edge-quantizer post-hoc dynamic int8 (quantize_ar.py drq8); repro = hf-to-litertlm audio8_tts_work/ (REPRODUCE entry) litert-torch 0.9.4 (~/venvs/lt094dev, the interpreter named in every export/quantize script's usage line: torch 2.13.0, transformers 5.14.1, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0 — pip show 2026-09-28; REPRODUCE entry: 'env with litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer'); PyTorch oracle transformers 4.57.6 / torch 2.12.1 CPU fp32 in ~/parakeet-env (FINDINGS §1) |
| **Command** | `SUFFIX=_v3 python export_fast.py (fast AR step graph, one signature `step`, 10-slot KV cache as I/O, fp32) -> python quantize_ar.py out/fast/fast_fp32_v3.tflite drq8 -> python verify_tflite_fast.py out/fast/fast_drq8_v3.tflite 3 -> python assemble_ship.py out/ship (copied as fast_ar_int8.tflite; built_from fast/fast_drq8_v3.tflite) (REPRODUCE entry)` |
| **Quantization** | dynamic int8 per-channel FULLY_CONNECTED + int8 EMBEDDING_LOOKUP, activations fp32 (quantize_ar.py drq8; card File table 'Dynamic int8'); teacher-forced vs oracle: fast logits max|d| 4.9, argmax 86% — each of the 4 blocks contributes ~2-3, i.e. the dynamic activation quantization, no single culprit; blockwise-32 int4 (max|d| 27, argmax 40%, ja speaker cosine 0.744 -> 0.691) is not shipped (FINDINGS §4-§5; NOTES) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `fast_ar_int8.tflite` | `79b47b0f38971b344ef40f7d88db48e0cde58d0a1ffe342d8919651e74436941` | 64.64 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 91.2% |
| **Partitions** | 10 |
| **Blocking ops** | LOGISTIC, DYNAMIC_UPDATE_SLICE, PACK, LESS, ADD, SELECT, RESHAPE, GATHER_ND, MAXIMUM, MINIMUM, EMBEDDING_LOOKUP |
| **Verdict provenance** | measured=258, unmatched=25 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-28/audio8-tts-preview-0.6b__fast_ar_int8.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2.48 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |
| webgpu_mldrift | output_mismatch | no | MISMATCH | 168.63736206304307 | 3.423 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-28 | measured |

## Pitfalls

- Step contract: inputs hidden [1,1,896] fp32, token [1] int32, use_hidden [1] fp32 (1.0 at position 0), pos [1] int32, mask [1,1,1,10] fp32 additive, k_all/v_all [4,1,2,10,64] fp32; outputs logits [1,4096], k_all, v_all. Called 10 times per frame: position 0 takes the slow hidden state, positions 1-9 the previous codebook token (card File table; FINDINGS §3). Bind buffers by the subgraph input order (hidden, token, use_hidden, pos, mask, k_all, v_all), not the alphabetical signature order, when using CompiledModel (NOTES 'Verified facts').
- 10 invokes per frame (~10 ms on the S26 CPU, 0.97 ms each); folding the 9 codebook steps into one graph with host-supplied noise would cut invoke overhead but not the 10x weight re-read (bandwidth-bound) (FINDINGS §8).
- Sampled model: quantized graphs sample a different but valid trajectory, so per-frame agreement with the reference is not a meaningful metric; the gate is the transcript (whisper large-v3-turbo WER/CER vs the input text) and the voice (TitaNet-L speaker cosine vs the reference clip) on 14 seeded sentences (6 en + 6 ja with a cloned voice, one of each without). PyTorch reference en WER 1.1% / ja CER 0.0% / cosine 0.66 / 0.74; int8 slow + int8 fast + fp16 codec 1.1% / 0.0% / 0.68 / 0.76; int4 slow 1.1% / 0.0% / 0.63 / 0.77; int8 codec on the reference codes 1.1% / 0.0% / 0.67 / 0.73. The one English error is shared with the reference (the model drops the first word of one sentence). The fp32 graphs reproduce the reference code sequence frame for frame on all 14 sentences (card Accuracy; FINDINGS §2, §5).
- The graphs are driven by a host loop (audio8_tts_litert.py, Python, ai-edge-litert Interpreter, CPU): prompt construction, chunked prefill, the vendor's sampler (temperature 0.7 / top-p 0.9 / top-k 50, repetition-aware re-draw), 10 fast-AR calls per frame, windowed codec decode, voice registration; on a phone the same graphs run from Kotlin/C++ through the CompiledModel API — no on-device functional run exists yet, the Galaxy S26 rows are per-graph benchmark_model latency + delegate coverage, functional parity is Mac-only on the same files (card Files + Performance; FINDINGS §8).
- Prompt length + generated frames must stay under 2,048 positions (the model's max_seq_len = the KV cache); the default cap is 512 frames (about 24 s) per call (card Limitations).
- GQA on the KV cache must not be expressed as repeat_interleave: it lowers to BROADCAST_TO (48 per step) and made decode 615 ms/step on the Mac (thread-count independent); folding the 7 query heads that share a kv head into the matmul row dimension removed it (16 ms/step fp32). RoPE: the vendor applies interleaved-pair rotation with a bf16-rounded table; the port permutes the q/k rows of wqkv to the rotate-half layout with the same bf16-rounded cos/sin as fp32 constants, bit-identical to the vendor buffers (FINDINGS §2).
- fp16 weights give the AR graphs no speed or RAM gain on the CPU (XNNPACK unpacks them to fp32 at init), so the AR graphs ship int8/int4 and only the codec ships fp16 (FINDINGS §4).
- Preview checkpoint: the vendor documents limited dialect coverage and sensitivity to noisy or mis-transcribed references. Generated speech can be misused for impersonation; obtain consent before cloning a voice and disclose synthetic audio (card Limitations).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
