---
family: vela-1.0-encoder-307m-embedding
license: apache-2.0
model_id: vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16
source_url: https://huggingface.co/litert-community/Vela-1.0-Encoder-307M-Embedding-LiteRT
task: sentence-similarity
---

# vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16

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
| **Quantization** | host-table form: fp16 FULLY_CONNECTED weights read through DEQUANTIZE; the token table stays outside the graph (tok_embeddings_fp16.bin, the checkpoint's float16 rows) and the host passes inputs_embeds (card 'Files', 'Conversion') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `vela_L128_embeds_wfp16_shift_arith.tflite` | `2713eab61b798e5d86362a25f7510dacfee109c55143bd25b7c53b4c04733ebd` | 210.791 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 92.1% |
| **Partitions** | 134 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1645, unmatched=45 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-18/vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 399.72 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-18 | measured |
| webgpu_mldrift | pass | yes | pass | 0.0028173834156779076 | 19.13 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-18 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-18/vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-18/vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16__pixel-8a.json`, `data/device_runs/2.2.0/2026-10-31/vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-31/vela-1.0-encoder-307m-embedding__vela_l128_embeds_fp16__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=DEFAULT (fp16); host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode); the SciFact texts of this window, scored as one retrieval run over the four window files | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-31 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP16_WITH_FP32_ACCUM; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 49.399167 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-18 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP16_WITH_FP32_ACCUM; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode); the SciFact texts of this window, scored as one retrieval run over the four window files | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-31 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 24.098672 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-18 | measured |
| galaxy-s26 | npu_qnn | pass | yes | - | sig:QualcommOptions JIT (HTP, burst), cold compile cache per run; host fp16 token table (inputs_embeds); C-API runner --accel npu+cpu | 13.062 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Qualcomm QNN HTP libraries (v81) in /data/local/tmp/vela_npu · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL SM-S942Q) | 2026-10-18 | measured |
| pixel-8a | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=DEFAULT (fp16); host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode); the SciFact texts of this window, scored as one retrieval run over the four window files | - | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-31 | measured |
| pixel-8a | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP16 (the delegate default); host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 67.639405 | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-18 | measured |
| pixel-8a | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP16_WITH_FP32_ACCUM; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 71.886922 | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-18 | measured |
| pixel-8a | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP16_WITH_FP32_ACCUM; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode); the SciFact texts of this window, scored as one retrieval run over the four window files | - | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-31 | measured |
| pixel-8a | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; host fp16 token table (inputs_embeds); gate app com.mlboydaisuke.d1omni.gate (generic mode) | 91.464844 | - | - | - | - | Pixel 8a · Google Tensor G3 (Mali-G715 GPU) · litert 2.2.0 · Android SDK 36 (Build.VERSION.SDK_INT read by the gate app; Build.MODEL Pixel 8a) | 2026-10-18 | measured |

## Pitfalls

- The LayerNorms of layers 12-21 and the final norm are computed on x*2^-6 (eps*2^-12), layers 9-11 on x*2^-2: the sum of squares of their inputs reaches 4e7 in float32, beyond float16's 65,504, and the rewrite keeps float16-storage GPU paths finite; exact in float32 (card 'Conversion').
- `inputs_embeds` float32 [1,128,768] (the host looks each token's row up in tok_embeddings_fp16.bin, the checkpoint's float16 table, 256,000 x 768) + `attention_mask` float32 [1,128] -> `embedding` float32 [1,768]; tokenize with the repo's tokenizer.json (`<bos>` = 2, `<eos>` = 1), pad the rows beyond the text with attention_mask 0 (card 'Files').
- The `_embeds_` files exist because of the GPU delegate: it does not take a float16 token table inside the graph, so this form takes the looked-up rows from the host and runs fully on the GPU (Metal: every op accelerated) and on the NPU; the pooling is written as multiply / sum / rsqrt because the delegate also refuses a masked select (`SELECT_V2` with a broadcast condition) (card 'Conversion').
- GPU fp16 arithmetic (the GPU defaults on Metal, Adreno and Mali) keeps every top-1 and the pair ranking but drops the per-row cosine floor to 0.991-0.998 on about 1 % of the texts, so the card's GPU numbers are at FP32; on the S26, FP16_WITH_FP32_ACCUM passes the full bar on L128-L512 at 84 ms (L512) if a 0.9994 floor is enough (card 'Performance').
- An int8 token table was tried and rejected: its per-row rounding error (0.75 % per row on average, 2.1 % on `<bos>`) moves short sentences by up to 0.7 % in cosine (floor 0.9929 over 3,918 texts) and no set of exactly-kept rows restores the 0.999 floor; int8 FC weights fail harder (floor 0.964); everything shipped is float16 (card 'Conversion').
- Every file is a static window: a text of n tokens (including `<bos>` and `<eos>`) runs in the smallest window with L >= n, and the time is the window's, not the text's; the NPU JIT compile on the S26 takes 4-34 s per window on the first run (cache it) (card 'Files', 'Performance').
- Strip the text before tokenizing: sentence-transformers does, and a trailing space adds a token (card 'Usage').
- The checks measure agreement with the publisher's own stack (cosine per text, top-1 retrieval, pair Spearman) on PAWS-X and STS-B v2 test rows, not accuracy against gold labels; the phone subsets are 140-497 texts per window, the Mac runs the full 3,918-4,184 (card 'Parity').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
