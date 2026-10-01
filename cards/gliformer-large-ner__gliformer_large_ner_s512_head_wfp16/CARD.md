---
family: gliformer-large-ner
license: apache-2.0
model_id: gliformer-large-ner__gliformer_large_ner_s512_head_wfp16
source_url: https://huggingface.co/litert-community/GLiFormer-Large-NER-LiteRT
task: token-classification
---

# gliformer-large-ner__gliformer_large_ner_s512_head_wfp16

| | |
|---|---|
| **Task** | token-classification |
| **Family** | gliformer-large-ner |
| **Source** | https://huggingface.co/litert-community/GLiFormer-Large-NER-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (torch 2.12.1; ai-edge-litert 2.1.6; ai-edge-quantizer 0.8.0 for the float16 weight storage; gliformer 0.1.2, gliner 0.2.29, transformers 5.16.1 — requirements-lock.txt / upstream.json) |
| **Command** | `not published as a script (card 'Provenance, conversion and license'): litert-torch 0.9.3 fixed-shape fp32 export of the NER path of knowledgator/gliformer-large-v1 (rev d0a4e53d) re-expressed for the GPU delegate without changing its math — host-side token lookup, one-hot routing as matmul, float masks, attention at rank 4, the exact DeBERTa logarithmic relative-position buckets as projected tables, the word BiLSTM unrolled for the window, one packed output tensor; then ai-edge-quantizer 0.8.0 float16 FLOAT_CASTING on the FULLY_CONNECTED weights (the wfp16 file); the layout and page embeddings of the backbone are skipped exactly as the upstream text path skips them` |
| **Quantization** | float16 weight storage (ai-edge-quantizer 0.8.0 FLOAT_CASTING on the FULLY_CONNECTED weights only): 9 float16 weight tensors read through a DEQUANTIZE to float32, every activation and every other constant float32; 25,103 operators in the fp32 reference (gliformer_large_ner_s512_head_fp32.tflite, 107,042,396 B, published as the exact reference) — 25,112 operators in this file; dynamic-range INT8 was not attempted — it does not compile on the LiteRT 2.2.0 GPU delegate for this graph family (card 'Provenance, conversion and license') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `gliformer_large_ner_s512_head_wfp16.tflite` | `7488194439006fae6433ffd4be090ca580905e179711cfb67356ac4c059183a3` | 53.874 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 79.6% |
| **Partitions** | 5128 |
| **Blocking ops** | DEQUANTIZE, LOGISTIC, TANH |
| **Verdict provenance** | measured=19994, unmatched=5118 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-26/gliformer-large-ner__gliformer_large_ner_s512_head_wfp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-26 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-26 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-26/gliformer-large-ner__gliformer_large_ner_s512_head_wfp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK (native runner, default threads) | 348.8584375 | - | - | - | 4679.66 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-09-26 | measured |

## Pitfalls

- The head graphs run on the CPU on Android (Accelerator.CPU): unrolled BiLSTM heads of this size crash the LiteRT 2.2.0 GPU compiler (see the split pitfall), so no GPU precision setting applies to this file; the encoder that feeds it needs GpuOptions(precision = FP32) — HOST_CONTRACT.md 'Numerical and memory limits': 'Explicit GPU FP32 computation is required; default GPU precision failed entity agreement' (card 'Kotlin'; HOST_CONTRACT.md).
- The graphs cover the encoder and the NER head only: the host looks up the word embeddings before the graph (host_assets/word_embeddings_fp32.bin, headerless little-endian float32 [128008,1024], 524,320,768 B, the default; or word_embeddings_fp16.bin, 262,160,384 B, upcast to float32 on lookup — gated at s128 only, longer-window fp16-table accuracy was not gated) and decodes the [1,1,T,15] start/end/inside logits with the upstream pairing decoder after it (threshold 0.5, flat spans). Build the encoded sequence with the pinned gliformer 0.1.2 processor ([SCHEMA] parent token, the five [ENTITY] label pairs, the separator, the text words; right-pad ids with 0 to N): a tokenizer call over a hand-concatenated prompt is not equivalent (HOST_CONTRACT.md 'Prompt, embeddings and routing').
- Bind the input buffers by name: the head's native buffer order is encoder_hidden [1,1,512,1024], text_routing [1,512,512], parent_routing [1,1,512], label_routing [1,5,512], text_mask [1,512] (the flatbuffer signature map sorts the names: encoder_hidden, label_routing, parent_routing, text_mask, text_routing); the head performs the text, parent and label routing matmuls internally and includes one bidirectional LSTM layer, anchor fusion and the span scorer; output_0 [1,1,512,15] reshapes to [1,512,5,3] with start, end, inside logits on the last axis (HOST_CONTRACT.md 's512'; graph_contract_s512.json).
- Fixed windows and a fixed class axis: N = 128 (T = 48 text words) / 256 / 512 encoded tokens including the complete label prompt, exactly five labels in a fixed order (the measured order is person, organization, location, product, date; other five-label sets are accepted by the API but were not numerically gated), batch 1. Inputs above the s512 limits raise an error naming both capacities; nothing is truncated or chunked implicitly — chunk_by_sentences(text, max_words) is a caller-side helper whose entity offsets are local to each chunk (card 'Host contract'; HOST_CONTRACT.md 'One extraction call').
- s256 and s512 ship as an encoder graph plus a head graph because the single s256 graph does not compile on the LiteRT 2.2.0 GPU: the checkpoint's word-level BiLSTM is unrolled for T steps and unrolled heads of 6,286 operators or more crash the GPU compiler (the head crashes at 570 MB RSS — a runtime defect reported with reproducers, not a memory limit), while the DeBERTa encoder alone (1,773 operators) compiles at every window. Create two CompiledModels — the encoder with Accelerator.GPU and FP32 precision, the head with Accelerator.CPU — and pass the encoder's [1,1,N,1024] output buffer to the head as its first input; the numbers are exact either way, only the placement differs (card 'What you get', 'Kotlin').
- Memory is the cost of this model (Galaxy S26, native CompiledModel processes without the token table, tokenizer or decoder): the 707 MB s128 graph is 2,591,805,440 B resident after loading and peaks at 4,542,996,480 B while the GPU delegate compiles it; the s256 encoder on the GPU with the fp32 head on the CPU in one process is 4,571,271,168 B resident; the s512 head alone is 4,676,505,600 B resident on the CPU — so treat s256 as the practical top window in an app, chunk longer documents by sentence, and plan for flagship-class phones; the token table adds 262 MB (fp16) or 524 MB (fp32) when resident (card 'What you get', memory table).
- Cold start: the first call after process start took 5,023 ms (compile 4,708 ms) for the s128 wfp16 graph in a fresh native process and 80 ms afterwards; the Android sample warms the whole pipeline 12 times before Ready and reported launch -> Ready 4,775 ms and a first Extract tap of 168 ms (fifth 155 ms) in a non-debuggable build with the screen on (card 'What you get', 'Kotlin', 'Measured quality and performance').
- Returned start / end are Unicode code-point offsets into the original Python string (end exclusive; text[start:end] is the entity text), not UTF-8 bytes and not Kotlin UTF-16 indices — translate them when supplementary characters occur (HOST_CONTRACT.md 'Output decoding').
- Validation scope: GPU execution was validated on the Galaxy S26 (SM-S942Q, SM8850 Adreno, Android 16, LiteRT 2.2.0) only — other Android GPU families, lower-memory phones and NPUs were not evaluated; English only, one text per call, the NER path only (classification, relation, structuring and embedding heads are not converted); the quality figures are agreement with the official gliformer 0.1.2 fp32 implementation (micro-F1 1.000, 70/70 identical span sets on desktop CPU; 10/10, 15/15, 20/20 on the phone), not a human-labelled accuracy estimate; latency is a single-device sample at battery 30-41 C with the phone idle — the s512 encoder number was taken at 41 C (card 'Measured quality and performance', 'Provenance, conversion and license').
- The s512 head is the largest resident graph of the package: 4,676,505,600 B on the CPU alone (4,727,980,032 B for the fp32 head, 8 % faster at 319.5 vs 348.9 ms median), and s512 simultaneous residency with its encoder was not measured — the card's advice is to treat s256 as the practical top window in an app and chunk longer documents by sentence (card 'What you get', 'Measured quality and performance'; HOST_CONTRACT.md 'Numerical and memory limits').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
