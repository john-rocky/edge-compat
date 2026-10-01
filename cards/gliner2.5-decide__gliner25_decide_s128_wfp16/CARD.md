---
family: gliner2.5-decide
license: apache-2.0
model_id: gliner2.5-decide__gliner25_decide_s128_wfp16
source_url: https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT
task: text-classification
---

# gliner2.5-decide__gliner25_decide_s128_wfp16

| | |
|---|---|
| **Task** | text-classification |
| **Family** | gliner2.5-decide |
| **Source** | https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (ai-edge-litert 2.1.6; torch 2.12.1; transformers 4.57.6; gliner2 2.0.0; ai-edge-quantizer 0.8.0 for the float16 weight storage) |
| **Command** | `cd conversion && python -B export_s128.py && python -B quantize_wfp16.py --seq 128  (conversion/README.md 'Order': the fp32 export first, then the weight-only float16 cast of that gated export; exports/gliner25_decide_s128_wfp16.tflite)` |
| **Quantization** | float16 weight storage (ai-edge-quantizer 0.8.0 FLOAT_CASTING, weight-only 16-bit): the 146 FULLY_CONNECTED weight tensors stored as float16 with a DEQUANTIZE to float32; activations and every other constant stay float32; 1,780 operators vs 1,634 in the fp32 reference graph; no INT8 file is shipped (card 'Files', 'Limits'; conversion/README.md) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `gliner25_decide_s128_wfp16.tflite` | `026a4b5eb62bb8f0110e9542fd4f788cf247bb5f6d3743a6056cad44b45fb39b` | 629.791 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 89.0% |
| **Partitions** | 196 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1731, unmatched=49 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-09-26/gliner2.5-decide__gliner25_decide_s128_wfp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-26 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-09-26 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-09-26/gliner2.5-decide__gliner25_decide_s128_wfp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:CPU XNNPACK 4 threads; Android sample app (debug build) | 213.804844 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-09-26 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32 | 69.639375 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-09-26 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; Android sample app (debug build) | 83.143021 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-09-26 | measured |

## Pitfalls

- Weight storage and GPU computation precision are separate settings: at the runtime's default GPU precision the s128 graph (float32 weights) compiled and returned finite logits on the Galaxy S26, but only 14 of 42 decisions matched the official result; with GpuOptions(precision = FP32) all 42 matched. Use the explicit FP32 option for every file (card 'GPU precision').
- The graph is the DeBERTa-v3-large encoder plus the classification head only: the host looks up the word embeddings before the graph (host_assets/word_embeddings_fp16.bin, [128011,1024] float16, upcast to float32 on lookup; the graph applies the embedding LayerNorm itself, so feed the raw rows) and turns the logits into decisions after it (softmax or sigmoid per task, thresholds). HOST_CONTRACT.md specifies the encoded sequence, the [L] marker positions and the decision rules (card 'Files'; HOST_CONTRACT.md).
- Assign the inputs by shape, not by name: the converter names them args_0..args_2 (inputs_embeds [1,N,1024], attention_mask [1,N] with 1.0 for encoded tokens, label_routing [1,32,N] one-hot at the j-th [L] marker); output_0 [1,1,1,32] holds one logit per label slot in request order, and slots past the label count hold a constant that is ignored (HOST_CONTRACT.md 'Graph signature').
- Fixed windows N = 128 / 256 / 512: the task schemas plus the text must fit in N encoded tokens with at most 32 labels in total; longer requests are rejected, never truncated, and gliner2's long-text chunking (classify_text_long) is not ported. Use the smallest window that holds the request (card 'Files', 'Limits').
- Back-to-back requests heat the phone: on the Galaxy S26 the GPU clock limit steps down from 1,300 MHz (s256 902 MHz, s512 646 MHz at the end of the job) and each s256 / s512 request gets slower during the job — the opening request is the cool value, the median mixes it with the throttled end (card 'Latency').
- GPU validation covers the Galaxy S26 (Adreno) only; NPU execution was not evaluated. No INT8 file is shipped: on GLiNER2.5 Small, the same family of graphs, dynamic-range INT8 FULLY_CONNECTED weights did not compile on the LiteRT 2.2.0 GPU, and no INT8 variant of this model was built or tested (card 'Limits').
- English only, as the source model; one text per call, no batching; few-shot examples and gliner2's extraction tasks (entities, relations, structures) are not part of the graphs (card 'Limits').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
