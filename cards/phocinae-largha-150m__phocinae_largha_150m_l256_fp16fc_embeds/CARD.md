---
family: phocinae-largha-150m
license: apache-2.0
model_id: phocinae-largha-150m__phocinae_largha_150m_l256_fp16fc_embeds
source_url: https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT
task: text-classification
---

# phocinae-largha-150m__phocinae_largha_150m_l256_fp16fc_embeds

| | |
|---|---|
| **Task** | text-classification |
| **Family** | phocinae-largha-150m |
| **Source** | https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (ai-edge-litert 2.2.0 for the desktop checks) |
| **Command** | `conversion/ scripts of the Hub repo (export.py per window, quantize.py for the fp16 FC weights and the table forms, tables.py for the host fp16 table); REPRODUCE.md of the Hub repo` |
| **Quantization** | host-lookup form: fp16 FULLY_CONNECTED weights read through DEQUANTIZE; the token table stays outside the graph (tables/tok_embeddings_fp16.bin, the publisher's float16 rows unchanged, [256000,384]) and the host passes inputs_embeds (card 'Conversion notes', 'Files') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `phocinae_largha_150m_L256_fp16fc_embeds.tflite` | `2ba31415bd13dbbdfec6f99241cedc3fc7539a68ffbaab5230953e3db404e4fa` | 88.387 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 91.7% |
| **Partitions** | 150 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1751, unmatched=50 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-11/phocinae-largha-150m__phocinae_largha_150m_l256_fp16fc_embeds.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 454.525 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.0033851218060208478 | 17.063 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-10/phocinae-largha-150m__phocinae_largha_150m_l256_fp16fc_embeds__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | output_mismatch | yes | MISMATCH | sig:CPU XNNPACK 4 threads; host fp16 token table (inputs_embeds); debug gate app com.phocinae.gate | 38.767083 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 36 build BP4A.251205.006.S942QOPS1AZH9 | 2026-10-10 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP16; host fp16 token table (inputs_embeds); debug gate app com.phocinae.gate | 15.331458 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 36 build BP4A.251205.006.S942QOPS1AZH9 | 2026-10-10 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP16_WITH_FP32_ACCUM; host fp16 token table (inputs_embeds); debug gate app com.phocinae.gate | 20.357761 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 36 build BP4A.251205.006.S942QOPS1AZH9 | 2026-10-10 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP32; host fp16 token table (inputs_embeds); debug gate app com.phocinae.gate | 28.867031 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 36 build BP4A.251205.006.S942QOPS1AZH9 | 2026-10-10 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:QualcommOptions JIT (HTP, BURST); host fp16 token table (inputs_embeds); debug gate app com.phocinae.gate | 16.793229 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 36 build BP4A.251205.006.S942QOPS1AZH9 | 2026-10-10 | measured |

## Pitfalls

- Every graph computes 21 LayerNorms on x·2⁻⁶ with the epsilon scaled to match: bit-identical in fp32; without it the sums of squares inside these norms leave the float16 range, so fp16 arithmetic and the NPU need it (card 'Conversion notes').
- Inputs: `input_ids` int32 [1,L] (int8emb) or `inputs_embeds` float32 [1,L,384] (embeds), `attention_mask` float32 [1,L], `qtype_onehot` float32 [1,3] in the order choice, score, noul; outputs `token_logits` float32 [1,L] and `pooled_cls` float32 [1,384]. Gather token_logits at each option's <mask> marker, divide by the question type's temperature from rl_agent_config.json (choice 0.8660, score 0.8081, noul 0.6625) and softmax; the act head takes pooled_cls + four features of the pre-temperature softmax and returns act_logits (card 'Raw CompiledModel contract').
- GPU FP16 (float16 arithmetic) passes the bar with less margin: it flipped 1 to 2 of the 20 near ties at L256 and its max |Δp| reaches 0.0188 at L512 (Mac Metal default, embeds); GPU FP32, FP16_WITH_FP32_ACCUM and the NPU keep max |Δp| at or below 0.0137 (card 'Limits').
- Dynamic int8 FULLY_CONNECTED fails the bar (max |Δp| 0.0835) and does not ship; the GPU delegate refuses a float16 table inside the graph, so the float16 table ships for host lookup instead (card 'Limits', 'Conversion notes').
- A graph computes every position of its window, so a smaller window runs faster; the host cuts the state to fit the window while the publisher's runtime cuts at 512 (card 'Conversion notes', 'Limits').
- The checks measure agreement with the publisher's fp32 runtime (phocinae-server) on the public typed-decisions test split, not accuracy against gold labels; measured on one Apple M4 Max and one Galaxy S26 only, not through the hfmodels-android SDK (card 'Checks', 'Limits').
- Of the 2,000 test decisions the state is cut on 1,378 at L256 (card 'Limits').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
