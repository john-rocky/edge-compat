---
family: internvl
license: apache-2.0
model_id: internvl3_5-1b
source_url: https://huggingface.co/OpenGVLab/InternVL3_5-1B
task: image-text-to-text
---

# internvl3_5-1b

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | internvl |
| **Source** | https://huggingface.co/OpenGVLab/InternVL3_5-1B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litertlm-convert fast_vlm pipeline, scripts/ship_internvl3_5_1b.sh) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `bash scripts/ship_internvl3_5_1b.sh` |
| **Quantization** | vision tower int8; Qwen3-0.6B decoder int4 blockwise-32 symmetric + OCTAV; input embedding int8 (externalized section) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `5ae4dbc96c8d4919e4776e9b8eee7f5ece8bdd2d1c057f73576c3d5f289d7ca4` | 780.13 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/internvl3_5-1b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/internvl3_5-1b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 202 | - | 222.23 | 26.13 | 950.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 962.12 | 43.45 | 230.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- GPU (Metal) fast_vlm path: a second image in the same conversation truncates the answer — ask about one image per chat on GPU; multi-image works on the CPU backend (also reproduces with litert-community/FastVLM-0.5B, so it is runtime-level, not model-level).
- Vision-only bundle (no audio tower): create the engine with the vision modality only — requesting the audio tower (.all) on a bundle with no audio section fails at session creation.
- Image input is resized to 448x448; ImageNet normalization and the NCHW transpose are baked into the vision encoder (the runtime feeds a [0,1] NHWC image).
- InternViT attention is rewritten 4D-clean (qkv split before the head reshape) to avoid a 5D intermediate for the GPU delegate.
- Decoder is extracted from the InternVLChat wrapper as a standalone Qwen3ForCausalLM with dynamic rope_scaling stripped; exported with cache <= base max so base RoPE is exact.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
