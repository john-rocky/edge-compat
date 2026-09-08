---
family: qwen3
license: apache-2.0
model_id: qwen3-reranker-0.6b
source_url: https://huggingface.co/litert-community/Qwen3-Reranker-0.6B-LiteRT
task: text-ranking
---

# qwen3-reranker-0.6b

| | |
|---|---|
| **Task** | text-ranking |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-Reranker-0.6B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python text-reranking/conversion/build_qwen3rerank.py (embedding table via export_embeddings.py)` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `963d91958f9aa3ac4b2bbd9ab8b848559c757025f80eb525123dbca7c68a23ac.tflite` | `d030d236b4f41c96bb62e3eca94a6252a74bf742a66b141124bab943a64b7c93` | 841.196 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/qwen3-reranker-0.6b.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-08-27/qwen3-reranker-0.6b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 78.494 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 47.508 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, AOT host-compiled, SM8850 target) · Android 16 | 2026-08-27 | measured |

## Pitfalls

- Token embedding lookup is host-side from the shipped embeddings_fp16.bin table; the graph consumes inputs_embeds [1,256,1024] and emits a baked 2-logit ('no'/'yes') head — host softmax gives P(yes) (zoo README L1852-1860).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
