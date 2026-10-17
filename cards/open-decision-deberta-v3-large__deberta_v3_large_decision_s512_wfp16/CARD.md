---
family: deberta-v3-large
license: apache-2.0
model_id: open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16
source_url: https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT
task: typed decisions (state + choice / score / noul questions -> one calibrated distribution
  per question; text-classification)
---

# open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16

| | |
|---|---|
| **Task** | typed decisions (state + choice / score / noul questions -> one calibrated distribution per question; text-classification) |
| **Family** | deberta-v3-large |
| **Source** | https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (torch 2.12.1; transformers 4.57.6; ai-edge-litert 2.1.6; ai-edge-quantizer 0.8.0 for the float16 weight storage) |
| **Command** | `python scripts/export.py --seq 512  (REPRODUCE.md section 3: fp32 export of the shaped-DeBERTa graph with the rank-4 matmul lowering, then the weight-only float16 cast of that export; exports/deberta_v3_large_decision_s512_wfp16.tflite)` |
| **Quantization** | float16 weight storage (ai-edge-quantizer 0.8.0 FLOAT_CASTING, weight-only 16-bit): the 146 FULLY_CONNECTED weight tensors stored as float16 with a DEQUANTIZE to float32; the 48 relative-position constants and all activations stay float32; 1,785 operators vs 1,639 in the fp32 export (card 'Files'); no INT8 file is shipped |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `deberta_v3_large_decision_s512_wfp16.tflite` | `238f0cc4dca7b6f18677950e9261d07401a3de9857d76f2a7450fada7f97a5d9` | 775.76 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 89.1% |
| **Partitions** | 196 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1736, unmatched=49 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-02/open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-02/open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-04/open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-05/open-decision-deberta-v3-large__deberta_v3_large_decision_s512_wfp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; Android sample app com.opendecision (zoo opendecision/, debug build, on-device tokenizer) | 619.77177 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; debug gate app com.decision.gate | 697.48927 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; hfmodels-android SDK device test (EncoderDecisions deberta_decision) | 1027.440625 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-04 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; hfmodels-android SDK device test (EncoderDecisions deberta_decision); parity step skipped (parity=0), the demo first after the warm-up | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-05 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; hfmodels-android SDK device test (EncoderDecisions deberta_decision); started after a 5-minute cool-down | 1031.236406 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-04 | measured |
| galaxy-s26 | npu_qnn | load_failed | - | - | sig:QualcommOptions htpPerformanceMode=BURST, JIT (no AOT); debug gate app com.decision.gate | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36, build BP4A.251205.006) | 2026-10-02 | measured |

## Pitfalls

- Run the GPU with explicit FP32 precision (GpuOptions precision = FP32 on Android, enforce_f32=True in Python). At the default precision every output is non-finite on Mac Metal (the attention mask constant -3.4e38 is -inf in fp16) and, on the Galaxy S26 with a finite-mask build, finite but wrong (216/274 answers).
- In fp16 computation two parts fail: the mask constant (NaN) and the LayerNorm sums of squares (1024 x 28.5^2 overflows fp16). The card documents the fp32-exact fix (-1e4 mask + SafeLayerNorm 2^-3); it is not applied to the shipped graphs, and the resulting NPU graph is not shipped (274/276 answers on the S26 Hexagon NPU at 110 ms).
- The S512 graph cannot be JIT-compiled for the Galaxy S26 NPU (Hexagon HTP prepare runs out of host memory 94 s in); the S256 graph compiles in 170-200 s as one DispatchDelegate node.
- Host contract: [CLS] [STATE] state[:256] ([Q] instructions ([OPT] option)+)+ [SEP] from the source tokenizer.json (keep the Precompiled charsmap; spm.model differs on exotic characters); the state is cut at 256 tokens, the sequence must fit the window (512) and the options the 128 slots; q_routing / o_routing rows are 1/len over the question / option text tokens; softmax at temperature 1.05 within each question.
- English only (banking77 / SST-5 / BoolQ training domains, per the author); the conversion reproduces the author's answers, including the wrong ones (author's in-domain accuracy 0.854, OOD 0.690).
- The S26 GPU warms under load: over 500 back-to-back 512-token requests the thermal status rose from 0 to 3 and per-request time from ~584 ms to ~1,024 ms (median 697.5 ms); the S256 graph measured 407.6 ms on an already warm phone.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
