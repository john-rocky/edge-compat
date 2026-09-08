---
family: llava-onevision
license: apache-2.0
model_id: llava-onevision-0.5b
source_url: https://huggingface.co/litert-community/LLaVA-OneVision-0.5B
task: image-text-to-text
---

# llava-onevision-0.5b

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | llava-onevision |
| **Source** | https://huggingface.co/litert-community/LLaVA-OneVision-0.5B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (LiteRT-LM fast_vlm bundle: SigLIP VISION_ENCODER [1,384,384,3]->[1,729,1152] + VISION_ADAPTER [1,729,1152]->[1,730,896] (projector + the learned image_newline token) + single-token EMBEDDER + PREFILL_DECODE) (HF card Conversion notes) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (not recorded in the HF card)` |
| **Quantization** | TODO — the HF card states no quantization recipe; decoder exported with the externalized (tied) embedder (HF card Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LLaVA-OneVision-0.5B.litertlm` | `7f7cd7ae3d2ae435a1f69651de02a9d51e916be60fdfef5cadc721725310971a` | 790.205 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/llava-onevision-0.5b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/llava-onevision-0.5b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 213 | - | 302.52 | 69.52 | 720.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 1376.31 | 86.45 | 160.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- Best for single-image VQA — one image per conversation: start a new conversation for a different image; single-image VQA produces coherent, image-grounded answers (CPU-verified) (HF card Quality + limitation).
- The vision encoder bakes OpenAI-CLIP normalization + the NCHW transpose into the graph; the single base-resolution (no-anyres) path is used so the image always maps to a fixed 730 soft tokens (HF card Conversion notes).
- License Apache-2.0 (LLaVA-OneVision + the Qwen2 language component) (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
