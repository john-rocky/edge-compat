---
family: modernbert-ja-310m
license: cc-by-sa-4.0
model_id: modernbert-ja-310m-decision__modernbert_ja_310m_decision_s256_wfp16
source_url: https://huggingface.co/mlboydaisuke/ModernBERT-Ja-310M-Decision-LiteRT
task: typed decisions in Japanese (question + state + candidates -> one logit per
  (context, candidate) pair; softmax over a question's candidates = choice / score
  / true-false; text-classification)
---

# modernbert-ja-310m-decision__modernbert_ja_310m_decision_s256_wfp16

| | |
|---|---|
| **Task** | typed decisions in Japanese (question + state + candidates -> one logit per (context, candidate) pair; softmax over a question's candidates = choice / score / true-false; text-classification) |
| **Family** | modernbert-ja-310m |
| **Source** | https://huggingface.co/mlboydaisuke/ModernBERT-Ja-310M-Decision-LiteRT |
| **License** | cc-by-sa-4.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (torch 2.12.1; transformers 4.57.6; ai-edge-litert 2.1.6; ai-edge-quantizer 0.8.0 for the float16 weight storage) |
| **Command** | `python scripts/export.py --seqs 256,512  (REPRODUCE.md: fp32 export of scripts/graph.py DecisionGraph at the fixed window, then the weight-only float16 cast; exports/modernbert_ja_310m_decision_s256_wfp16.tflite)` |
| **Quantization** | float16 weight storage (ai-edge-quantizer 0.8.0 FLOAT_CASTING, weight-only 16-bit on FULLY_CONNECTED): the 102 FULLY_CONNECTED weight tensors stored as float16 with a DEQUANTIZE to float32; the RoPE and band constants and all activations stay float32; 1,911 operators vs 1,809 in the fp32 export; no INT8 file is shipped |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `modernbert_ja_310m_decision_s256_wfp16.tflite` | `57a2ac7254dd8c8d08efad9d543a2e770bf1b16eaf2b8311653160f91caf50c1` | 451.936 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 91.9% |
| **Partitions** | 155 |
| **Blocking ops** | DEQUANTIZE, SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=1859, unmatched=52 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-02/modernbert-ja-310m-decision__modernbert_ja_310m_decision_s256_wfp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | run_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-02/modernbert-ja-310m-decision__modernbert_ja_310m_decision_s256_wfp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; debug gate app com.decisionja.gate | 229.156823 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:QualcommOptions htpPerformanceMode=BURST, JIT (first compile, no cache); debug gate app com.decisionja.gate | 41.726354 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | npu_qnn | output_mismatch | yes | MISMATCH | sig:QualcommOptions htpPerformanceMode=BURST, JIT from the on-device cache; debug gate app com.decisionja.gate; float32 host table (not published) | 44.617136 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |

## Pitfalls

- Run the GPU with explicit FP32 precision (GpuOptions precision = FP32 on Android, enforce_f32=True in Python). The shipped graphs carry two fp32-exact rewrites (mask constant -1e4; SafeLayerNorm 2^-k, k 5-6 in layers 14-24 and 2 on the final norm, because the residual reaches 2,694), so fp16 computation is finite, but it still moves answers: Mac Metal default precision keeps 619 of 621 winners (32 questions over 0.01, max 0.042); the Galaxy S26 GPU at default precision or FP16 keeps all 60 winners of a 60-request probe with 2 questions over 0.01 (max 0.0202); FP16_WITH_FP32_ACCUM stays within 0.0045 on those 60 but runs as slowly as explicit FP32.
- The Galaxy S26 Hexagon NPU (JIT, BURST) compiles both windows whole (29.5 s at S256, 83.0 s at S512) and keeps every winning candidate on 618 / 621 questions, but one question moves by 0.0113 (bar 0.01; the same with a float32 host table, so it is the HTP's fp16 arithmetic). The card lists the NPU with that number; the verified path is the GPU with explicit FP32. NPU warm medians 41.7 ms per 256-token pair and 142.6 ms per 512-token pair against 229.2 and 416.4 ms on the GPU with explicit FP32 on the same hot phone.
- Host contract (the author's serving code): context = '質問: {instructions}\n状況: {state}' (a non-string state is json.dumps(ensure_ascii=False)); candidate = '{label}' or '{label} — {description}' (U+2014 with one space each side); choice labels = criteria keys, score labels = '0'..'K-1', noul labels = 'true' / 'false'; one pair per candidate '<s> context </s><s> candidate </s>' from tokenizer.json with truncation='only_first' at 512 (the context end is cut, never the candidate); pad id 3; softmax at temperature 1 over a question's candidates; confidence = 1 - H(p)/ln K.
- transformers 4.57.6 cannot load this repository's tokenizer class (TokenizersBackend, saved by transformers 5.17.0); the tokenizers library on tokenizer.json gives the reference ids on all 2,357 fixture pairs, and so does the base model's tokenizer.json.
- The checkpoint is F32, so the float16 token table is a rounding (max 6.1e-5, 104 of 78.6M values flush to zero): fp32 graph + float16 table differs from the reference by up to 0.00039 in probability, the shipped float16 weights + float16 table by 0.00079, every winner kept on all 621 questions.
- One graph call per candidate: a request costs one call per candidate across its questions (the 547 fixture requests took 2,357 calls); the S256 graph raises on a pair longer than 256 tokens (546 of the 547 fixture requests fit).
- The S26 GPU warms under load: the S512 explicit-FP32 run started at thermal status 0 (34.9 C) and ended at status 2 (44.8 C) after 17 minutes, per-pair time from 183.6 ms (minimum) to 669.0 ms (maximum, median 416.4 ms); every later row started hot.
- Japanese; per the author only Japanese and English were evaluated, business domains such as customer support were not in the training data, and wrong answers can come with high probability. The conversion reproduces the reference's answers, including the wrong ones (our JGLUE test subset: JNLI 238/250, JCommonsenseQA 233/250; the author's 16 items 14/16).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
