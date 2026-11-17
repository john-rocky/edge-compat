---
family: vela-1.0-encoder-307m-embedding
license: apache-2.0
model_id: vela-1.0-encoder-307m-embedding__vela_l64_fp16
source_url: https://huggingface.co/litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT
task: sentence-similarity
---

# vela-1.0-encoder-307m-embedding__vela_l64_fp16

| | |
|---|---|
| **Task** | sentence-similarity |
| **Family** | vela-1.0-encoder-307m-embedding |
| **Source** | https://huggingface.co/litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (litert-converter 0.5.0.dev20260912; ai-edge-litert 2.2.0 for the desktop checks) |
| **Command** | `the conversion scripts and the full record are in github.com/john-rocky/hf-to-litertlm vela_work/ (card 'Licence and source'); the card names the tools and the graph, not a command line` |
| **Quantization** | one-file form: fp16 FULLY_CONNECTED weights and the fp16 token table read through DEQUANTIZE inside the graph (ai-edge-quantizer weight-only; activations float32); int8 weights were tried and rejected (card 'Conversion') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `vela_L64_ids_v3_fp16fc_fp16emb_shift_arith.tflite` | `4703af6896075ea8d30ea5b880f1f91a5f3b90036ad57455ccb34bdf0322cbd7` | 585.639 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 91.9% |
| **Partitions** | 134 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE, RESHAPE, EMBEDDING_LOOKUP |
| **Verdict provenance** | measured=1642, unmatched=47 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-18/vela-1.0-encoder-307m-embedding__vela_l64_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-18/vela-1.0-encoder-307m-embedding__vela_l64_fp16__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK 4 threads; fp16 token table in the graph (input_ids); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 19.8360675 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-18 | measured |
| galaxy-s26 | npu_qnn | pass | yes | - | sig:QualcommOptions JIT (HTP, burst), cold compile cache per run; fp16 token table in the graph (input_ids); C-API runner --accel npu+cpu | 6.823 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Qualcomm QNN HTP libraries (v81) in /data/local/tmp/vela_npu · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-18 | measured |
| pixel-8a | cpu_xnnpack | fallback | no | pass | sig:CPU XNNPACK 4 threads; fp16 token table in the graph (input_ids); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 180.72762 | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-18 | measured |
| pixel-8a | gpu_mldrift | run_failed | - | - | sig:GpuOptions precision=FP16 (the delegate default); fp16 token table in the graph (input_ids); gate app com.mlboydaisuke.d1omni.gate (generic mode) | - | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-18 | measured |

## Pitfalls

- The LayerNorms of layers 12-21 and the final norm are computed on x*2^-6 (eps*2^-12), layers 9-11 on x*2^-2: the sum of squares of their inputs reaches 4e7 in float32, beyond float16's 65,504, and the rewrite keeps float16-storage GPU paths finite; exact in float32 (card 'Conversion').
- `input_ids` int32 [1,64] + `attention_mask` float32 [1,64] -> `embedding` float32 [1,768] (the normalized vector); tokenize with the repo's tokenizer.json (its post-processor adds `<bos>` = 2 and `<eos>` = 1), pad the rows beyond the text with attention_mask 0 and input_ids 0 (card 'Files').
- The one-file form is the CPU form (and the NPU takes it whole): the GPU delegate does not take a float16 token table inside the graph (`EMBEDDING_LOOKUP: Empty quantization params`); the `_embeds_` files of the same repo take the looked-up rows from the host and run fully on the GPU and the NPU (card 'Conversion').
- GPU fp16 arithmetic (the GPU defaults on Metal, Adreno and Mali) keeps every top-1 and the pair ranking but drops the per-row cosine floor to 0.991-0.998 on about 1 % of the texts, so the card's GPU numbers are at FP32; on the S26, FP16_WITH_FP32_ACCUM passes the full bar on L128-L512 at 84 ms (L512) if a 0.9994 floor is enough (card 'Performance').
- An int8 token table was tried and rejected: its per-row rounding error (0.75 % per row on average, 2.1 % on `<bos>`) moves short sentences by up to 0.7 % in cosine (floor 0.9929 over 3,918 texts) and no set of exactly-kept rows restores the 0.999 floor; int8 FC weights fail harder (floor 0.964); everything shipped is float16 (card 'Conversion').
- Every file is a static window: a text of n tokens (including `<bos>` and `<eos>`) runs in the smallest window with L >= n, and the time is the window's, not the text's; the NPU JIT compile on the S26 takes 4-34 s per window on the first run (cache it) (card 'Files', 'Performance').
- Strip the text before tokenizing: sentence-transformers does, and a trailing space adds a token (card 'Usage').
- The checks measure agreement with the publisher's own stack (cosine per text, top-1 retrieval, pair Spearman) on PAWS-X and STS-B v2 test rows, not accuracy against gold labels; the phone subsets are 140-497 texts per window, the Mac runs the full 3,918-4,184 (card 'Parity').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
