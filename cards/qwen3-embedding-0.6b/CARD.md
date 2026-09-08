---
family: qwen3
license: apache-2.0
model_id: qwen3-embedding-0.6b
source_url: https://huggingface.co/litert-community/Qwen3-Embedding-0.6B-LiteRT
task: sentence-similarity
---

# qwen3-embedding-0.6b

| | |
|---|---|
| **Task** | sentence-similarity |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-Embedding-0.6B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python text-embedding/conversion/build_qwen3emb.py (embedding table via export_embeddings.py)` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `00519d5f6f83208e8a35817fc827a234e8070ed07dcb967aca06c40129bf4c21.tflite` | `6025e85bf481b9fb370dcd76b6ff5154d8958a80ab0e91521c3d1babf0c218f7` | 840.879 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/qwen3-embedding-0.6b.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | load_failed | - | - | - | - | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-08-27/qwen3-embedding-0.6b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 44.738 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 25.44 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, AOT host-compiled, SM8850 target) · Android 16 | 2026-08-27 | measured |

## Pitfalls

- Token embedding lookup is host-side from the shipped embeddings_fp16.bin table (GATHER is GPU-banned); the graph consumes inputs_embeds [1,128,1024] (zoo README L1830-1831; HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
