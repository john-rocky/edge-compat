---
family: gliner2.5-multi
license: apache-2.0
model_id: gliner2.5-multi__gliner25_multi_s512_wfp16
source_url: https://huggingface.co/litert-community/GLiNER2.5-Multi-LiteRT
task: token-classification
---

# gliner2.5-multi__gliner25_multi_s512_wfp16

| | |
|---|---|
| **Task** | token-classification |
| **Family** | gliner2.5-multi |
| **Source** | https://huggingface.co/litert-community/GLiNER2.5-Multi-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (ai-edge-litert 2.1.6; torch 2.12.1; transformers 4.57.6; gliner2 2.0.0; ai-edge-quantizer 0.8.0 for the float16 weight storage) |
| **Command** | `export_round3.py --seq 512 (unified dense-prefix recipe with --mask-neg -1e4 --ln-shift 3) then quantize_shipment.py --seq 512 (ai-edge-quantizer 0.8.0 FLOAT_CASTING, FULLY_CONNECTED weights only); graph contract graph_contract_s512.json` |
| **Quantization** | float16 weight storage (ai-edge-quantizer 0.8.0 FLOAT_CASTING, weight-only 16-bit): the 96 FULLY_CONNECTED weight tensors stored as float16 with a DEQUANTIZE to float32; inputs, outputs, activations and every other constant stay float32 (card 'Files and supported configuration'). |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `gliner25_multi_s512_wfp16.tflite` | `ee4bd32572626e6a256a44954622cc2b610bfd99df6eb1311a44952cd12a04c6` | 238.655 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 89.0% |
| **Partitions** | 127 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE, LOGISTIC |
| **Verdict provenance** | measured=1119, unmatched=30 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-02/gliner2.5-multi__gliner25_multi_s512_wfp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |
| webgpu_mldrift | pass | yes | - | - | 50.885 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-02/gliner2.5-multi__gliner25_multi_s512_wfp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | output_mismatch | no | MISMATCH | sig:CpuOptions numThreads=4 (XNNPACK); debug gate app com.gliner25multi.gate | 407.261745 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | output_mismatch | yes | MISMATCH | sig:GpuOptions precision=FP32; debug gate app com.gliner25multi.gate | 422.52502549999997 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:QualcommOptions htpPerformanceMode=BURST, JIT (first compile, cache dir of a fresh app process); debug gate app com.gliner25multi.gate | 186.5391665 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |

## Pitfalls

- Weight storage does not set the GPU compute precision: request GpuOptions(precision = FP32). At the default precision the outputs stay finite but one of 70 span sets changes (card 'Measured quality and performance', s128 row 69/70, max confidence drift 3.3e-2).
- The graph is the dense prefix of the upstream BoundaryExtractor (mDeBERTa-v3-base encoder, boundary encoder, boundary query head, per-token projections): the host looks up the word-embedding rows before the graph (host_assets/word_embeddings_fp16.bin [250112,768] float16, rows upcast to float32; fp32 table as reference) and runs the upstream gliner2 sparse decoder after it (host_assets/runtime/, 16 decoder tensors); the five inputs are args_0..args_4 = inputs_embeds [1,512,768], attention_mask [1,512], text_routing [1,384,512], query_routing [1,5,512], text_mask [1,384]; one packed float32 output [1,1,1,579422] with 17 logical slices (graph_contract_s512.json).
- The caller selects the word splitter per request — official whitespace for space-delimited text such as English, official char for Japanese / CJK; there is no language detector, and capacity counts processed text slots and encoded tokens (text ending in 。 takes one extra slot). Fixed window N=512 / T=384: longer inputs are rejected, not truncated (card 'Files and supported configuration', HOST_CONTRACT.md).
- Two fp32-exact rewrites are inside the shipped graph (attention-mask fill -10000 instead of the float32 minimum; each of the 29 active LayerNorms computed on x*2^-3 with eps*2^-6; 60/60 bit-identical comparisons with the stock graph) so the same files stay finite under fp16 delegates — finite, not exact: the Galaxy S26 Hexagon NPU (JIT) matched 69/70, 72/75 and 77/80 span sets with every difference a 0.5-threshold near-tie (max confidence drift 4.8e-2), so the card reports the NPU and does not offer it as a supported path (card 'Measured quality and performance').
- Back-to-back requests heat the phone: the s256 and s512 GPU jobs rose from thermal status 0 to 1-2 and their sustained medians (93.0 / 422.5 ms) are 1.5-2.1x the cool opening values (63.5 / 200.9 ms); the card states both with their condition. Latency is a single-device sample (card 'Measured quality and performance').
- Validated only for the five fixed labels in that order, threshold 0.5, English (whitespace) and Japanese (char), one window per call, on the Galaxy S26 (card 'Limits'); conversion fidelity to the official gliner2 fp32 implementation (span F1 1.000 on 80 bilingual inputs), not a general accuracy claim — upstream misses carry over (4 of 10 expected organizations on ten early Japanese sentences).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
