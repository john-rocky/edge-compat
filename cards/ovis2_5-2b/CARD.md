---
family: ovis
license: apache-2.0
model_id: ovis2_5-2b
source_url: https://huggingface.co/AIDC-AI/Ovis2.5-2B
task: image-text-to-text
---

# ovis2_5-2b

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | ovis |
| **Source** | https://huggingface.co/AIDC-AI/Ovis2.5-2B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litertlm-convert fast_vlm pipeline, scripts/ship_ovis_2b.sh) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `bash scripts/ship_ovis_2b.sh` |
| **Quantization** | vision tower int8; Qwen3-1.7B decoder int4 blockwise-32 symmetric + OCTAV; input embedding int8 (externalized section) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ovis2.5-2B.litertlm` | `528485444515cef66d999dfa551be5740854612cf9bfc1003ff5b9c7281afc7b` | 2049.657 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/ovis2_5-2b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 461.75 | 26.83 | 480.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- Reasoning VLM: the model may emit a <think>...</think> block before its final answer (matches the base model) — allow enough max-tokens (>=1024) for the answer to follow.
- Vision-only bundle (no audio tower): bring up the engine with the vision modality only — requesting the audio tower (.all) on a bundle with no audio section fails at session creation.
- The dynamic-resolution Siglip2-NaViT tower does not torch.export; shipped as a static 512x512 rewrite (valid because fullatt_block_indexes=None makes the window-reorder a no-op) — static-vs-original feature corr 0.99999964.
- Siglip normalization ((x-0.5)/0.5) is baked into the encoder; the runtime feeds a [0,1] NHWC image resized to 512x512.
- Ovis's two learned image-boundary indicator embeddings are omitted (the fast_vlm path splices only the atom embeddings) — verified to stay coherent in eager.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
