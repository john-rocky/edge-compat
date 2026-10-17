---
family: gliclass-edge-v3.0
license: apache-2.0
model_id: gliclass-edge-v3.0__gliclass_edge_v3_s256_fp32
source_url: https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT
task: zero-shot-classification
---

# gliclass-edge-v3.0__gliclass_edge_v3_s256_fp32

| | |
|---|---|
| **Task** | zero-shot-classification |
| **Family** | gliclass-edge-v3.0 |
| **Source** | https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (torch 2.12.1; transformers 5.17.0; gliclass 0.1.20; ai-edge-litert 2.1.6; ai-edge-quantizer 0.8.0 for the float16 weight storage) |
| **Command** | `conversion/export.py --seq 256 --matmul rank4 --ln-shift audit --suffix sln (litert-torch convert of the classification-path graph; the fp16-weight file by ai-edge-quantizer FLOAT_CASTING in the same script); card 'Provenance, conversion and license', REPRODUCE.md of the Hub repo` |
| **Quantization** | float32 weights: 670 operators, no quantization (card 'Files': the default download). |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `gliclass_edge_v3_s256_fp32.tflite` | `aff0ee9795a2ffc2d4d32385e8f804b30bac4282d96d19c1e53cac88dcc97708` | 51.465 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 96.9% |
| **Partitions** | 22 |
| **Blocking ops** | SQUARED_DIFFERENCE |
| **Verdict provenance** | measured=649, unmatched=21 |
| **Lint report schema** | 1.1 |

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-10-02/gliclass-edge-v3.0__gliclass_edge_v3_s256_fp32.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 101.72 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |
| webgpu_mldrift | pass | yes | pass | 1.636965062852597e-05 | 7.918 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-10-02 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-02/gliclass-edge-v3.0__gliclass_edge_v3_s256_fp32__galaxy-s26.json`, `data/device_runs/2.2.0/2026-10-04/gliclass-edge-v3.0__gliclass_edge_v3_s256_fp32__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | - | sig:CpuOptions 4 threads; hfmodels-android SDK device test (EncoderDecisions gliclass) | 24.105937 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-04 | measured |
| galaxy-s26 | cpu_xnnpack | pass | yes | - | sig:CpuOptions numThreads=4 (XNNPACK); Android sample app com.gliclass (zoo gliclass/, debug build, on-device tokenizer) | 15.649375 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; Android sample app com.gliclass (zoo gliclass/, debug build, on-device tokenizer) | 8.230103999999999 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | pass | sig:GpuOptions precision=FP32; debug gate app com.gliclass.gate | 8.322448 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (SDK 36) | 2026-10-02 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP32; hfmodels-android SDK device test (EncoderDecisions gliclass) | 10.000468 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 (build BP4A.251205.006.S942QOPS1AZH9) | 2026-10-04 | measured |

## Pitfalls

- Weight storage does not set the GPU compute precision: request GpuOptions(precision = FP32). At the default precision the outputs stay finite but the answers move: Galaxy S26 475/482 top labels and 455/482 label sets, Mac Metal 479/469 (card 'Galaxy S26, gate app' and 'fp16, the GPU default precision and the NPU').
- The graph takes host-built inputs: inputs_embeds [1,256,384] from host_assets/tok_embeddings_fp16.bin ([50370,384] float16 rows upcast to float32, [PAD] 50283 rows at padded positions), attention_mask [1,256] (1 real / 0 pad), label_routing [1,25,256] (row k = 1 at the k-th <<LABEL>> token); output logits [1,1,1,25], slots past the request's labels are ignored; softmax (single-label) or sigmoid >= threshold (multi-label) runs on the host (card 'Host contract').
- The request string is <<LABEL>>label 1<<LABEL>>label 2…<<SEP>> + prompt + text with nothing between the prompt and the text; tokenize with the source tokenizer.json ([CLS] … [SEP], ByteLevel prefix space per piece between added tokens); at most 25 labels and 256 tokens, longer requests raise an error, nothing is truncated (card 'Host contract', 'Limits').
- The shipped graph computes 15 LayerNorms on x*2^-k with eps*2^-2k (k = 5 in layers 3-6, 6 in layers 7-9 and the final norm) and uses a -1e4 mask; this is exact in FP32 (bit-identical logits to the unmodified graph) and keeps fp16 computation finite but not exact: the Galaxy S26 Hexagon NPU (JIT, one partition, 2.51 ms) keeps 480/482 top labels and 462/482 label sets, every change a near tie, so the card reports the NPU and does not call it a validated path (card 'fp16, the GPU default precision and the NPU').
- The checkpoint is float32: the float16 host table rounds values by up to 1.2e-4 (max logit difference 0.0090 against 0.000036 with a float32 table, same answers on all 552 requests) (card 'fp16 …').
- Validated on one desktop (Apple M4 Max, ai-edge-litert 2.1.6 CPU and Metal explicit FP32) and one phone (Galaxy S26, LiteRT 2.2.0 GPU explicit FP32 and CPU); English only; conversion fidelity to the official gliclass pipeline (552/552 answers), not a task-accuracy claim — own-subset gold agreement 67/144 SemIf, 115/200 ag_news, 71/200 banking77 (25-way) is not the publisher's benchmark (card 'Agreement with the dataset labels', 'Limits').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
