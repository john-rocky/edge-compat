---
family: granite-docling
license: apache-2.0
model_id: granite-docling-258m
source_url: https://huggingface.co/ibm-granite/granite-docling-258M
task: image-text-to-text
---

# granite-docling-258m

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | granite-docling |
| **Source** | https://huggingface.co/ibm-granite/granite-docling-258M |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch fast_vlm rail (SmolVLM2 vision scripts, docling_work/); LiteRT-LM bundle: SigLIP-base p16 512x512 vision encoder int8 (64 image tokens after pixel-shuffle x4) + granite Llama-architecture 576-dim/30-layer decoder hf-to-litertlm docling_work @ d18f276 (SmolVLM2 rail, 512-BILINEAR contract) |
| **Command** | `docling_work/ship_granite_docling.sh (see hf-to-litertlm REPRODUCE.md VLM table row granite-docling-258m)` |
| **Quantization** | decoder int8 weights with FLOAT compute (integer-compute int8/int4 corrupt DocTags structure — measured, int4 rejected); vision tower int8 (corr 0.98 vs fp32, structurally identical output); cache 4096 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-docling-258M.litertlm` | `11a14f547dda752d35c222b2a781d277fa499d94d14a3b575f719bb16391e113` | 322.291 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-05/granite-docling-258m__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/granite-docling-258m__pixel-8a.json`, `data/device_runs/0.16.1/2026-08-24/granite-docling-258m__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-08-24 | measured |
| galaxy-s26 | cpu | fallback | no | - | 204 | - | 313.36 | 33.49 | 680.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | load_failed | - | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | fallback | no | - | 204 | - | 88.72 | 23.25 | 2340.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| pixel-8a | gpu | run_failed | yes | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Input contract: pre-resize the page to exactly 512x512 with BILINEAR resampling in the app before sending. The runtime's internal resampler uses a different filter and the model then hallucinates a page (base-model trait: single-global-512 path is resampling-filter-sensitive).
- CPU-only ship: GPU delegation is possible but ~4x slower than CPU on this 258M decoder (dispatch-bound), and the shipped wi8-float form's DEQUANTIZE->FULLY_CONNECTED pattern is rejected by GPU delegates.
- The newline after <|end_of_text|> is part of the chat format — dropping that single token turns output into a hallucinated blank page. If you re-template, protect the \n.
- Dense multi-column pages and small formulas degrade in single-512 mode (also in the original model run this way) — tile the page app-side and send crops for those.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
