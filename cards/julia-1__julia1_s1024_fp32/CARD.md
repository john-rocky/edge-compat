---
family: julia-1
license: apache-2.0
model_id: julia-1__julia1_s1024_fp32
source_url: https://huggingface.co/litert-community/Julia-1-LiteRT
task: text-classification
---

# julia-1__julia1_s1024_fp32

| | |
|---|---|
| **Task** | text-classification |
| **Family** | julia-1 |
| **Source** | https://huggingface.co/litert-community/Julia-1-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (torch 2.12.1; transformers 5.0.0; ai-edge-litert 2.1.6 for the desktop checks) |
| **Command** | `conversion/ scripts of the Hub repo (host-side token lookup, rotary tables baked per window, sliding-window band as a constant, question type as a one-hot matmul, the two head layers written out, exact GELU); REPRODUCE.md of the Hub repo` |
| **Quantization** | float32 weights: 1,704 operators, no quantization; the token table is the checkpoint's float32 embedding rounded to float16 on the host side (card 'Files', 'Provenance'). |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `julia1_s1024_fp32.tflite` | `b4f260595776c0a126f0e1aadbefb0c2acc6e0b1ca0cfe9499b4d322d591e632` | 179.771 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 97.1% |
| **Partitions** | 51 |
| **Blocking ops** | SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1654, unmatched=50 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-09/julia-1__julia1_s1024_fp32.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2014.363 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-09 | measured |
| webgpu_mldrift | pass | yes | pass | 0.006589250508776393 | 43.43 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-09 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-30/julia-1__julia1_s1024_fp32__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; host fp16 token table (shipped form); debug gate app com.julia.gate | 298.463958 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-09-30 | measured |

## Pitfalls

- Weight storage does not set the GPU compute precision: request GpuOptions(precision = FP32). The fp16 paths change answers: S26 Hexagon NPU (JIT, fp32 graph) 685/706 same argmax with probabilities off by up to 0.418; Mac Metal at its default fp16 precision 375/400 (card 'fp16 and the NPU', 'Measured quality and performance').
- The graph takes host-built inputs: inputs_embeds [1,1024,384] from julia1_token_table_fp16.bin ([256000,384] little-endian float16 rows, padding positions included), attention_mask [1,1024] (1 real / 0 pad), qtype_onehot [1,3] (choice / score / noul); output token_logits [1,1024], read at the <mask> marker positions; softmax at temperature 1 runs on the host (card 'Host contract').
- Sequence: <bos> '{type} question: {question}' <eos> <mask> ' option0' <mask> ' option1' … <eos> state <eos>; 2 to 20 options, at most 48 tokens each; a request that does not fit raises an error, nothing is truncated (35 of the 2,000 typed-decisions questions need more than 512 tokens and fit the S1024 graph) (card 'Host contract').
- The graph computes LayerNorm on x*2^-k with eps*2^-2k (k = 8 in layers 12-21 and the final norm, k = 2 in the head and scorer norms), uses -1e4 attention masks and scales q by 1/8 before QK^T; exact in fp32 (bit-identical token logits), finite but not the same answers in fp16 (card 'fp16 and the NPU', 'Provenance').
- The float16 token table changes no answer and moves probabilities by up to 0.0077 (desktop CPU and S26 GPU); the host lookup is outside the measured graph time (Kotlin loop 19.0 ms float32 table / 23.6 ms float16 table per question in the gate app) (card 'Files', 'Measured quality and performance').
- Validated against the author's runtime (CPU FP32), not task accuracy; one phone (Galaxy S26 SM-S942Q, LiteRT 2.2.0 Kotlin CompiledModel); English text only; S26 GPU explicit FP32: 100/100 rows (the 35 typed-decisions questions longer than 512 tokens + 65 others) with the shipped float16 table, max probability difference 0.0077, median 298.5 ms at thermal status 2; desktop CPU all 2,000 typed-decisions questions with the author's answers (426/600 choice, 542/800 score, 483/600 noul) (card 'Limits', 'Measured quality and performance').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
