---
family: smolvlm
license: apache-2.0
model_id: smolvlm2-500m
source_url: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
task: image-text-to-text
---

# smolvlm2-500m

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | smolvlm |
| **Source** | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | TODO (owner) — the card describes the fast_vlm bundle layout and graph edits but does not name the converter tool TODO (owner) — not recorded in the source card |
| **Command** | `TODO (owner) — not recorded in the source card` |
| **Quantization** | vision: SigLIP encoder + pixel-shuffle x4 + Linear connector int8 -> 64 image tokens; decoder SmolLM2-360M int4 weights (blockwise-32 + OCTAV); tied embedding INT8 (externalized); integer compute; KV cache 2048. File SmolVLM2-500M.litertlm, ~361 MB |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `SmolVLM2-500M.litertlm` | `0dfb6fb881eb16e5ef2b2be04de5476caf939b7d9ae601fdee308bbc5462fd55` | 344.326 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/smolvlm2-500m__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/smolvlm2-500m__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 203 | - | 417.03 | 76.72 | 500.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 204 | - | 1371.43 | 76.73 | 160.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- One image per conversation on GPU: a second image in the same conversation may degrade — a GPU-delegate trait shared across fast_vlm models. CPU handles multi-image; start a new conversation for a different image.
- Gallery import: in the Import Model dialog you must check "Support image" or image input will not work; Gallery v1.0.16+ can also import directly from Hugging Face inside the app.
- Vision-only bundle, no audio tower — on the Swift runtime load with the vision tower enabled (Modality.textImage / [.vision]).
- Very small (500M) model: keep a sensible max_tokens and use sampling (e.g. top-p); at pure greedy it can be repetitive/verbose.
- Image input is resized to 512x512, with the (x-0.5)/0.5 normalization and NCHW transpose baked into the vision-encoder graph.
- The vision encoder uses the static arange(1024) position-embedding path — the model's dynamic bucketize position logic is bypassed, numerically identical only for a full 512x512 frame. Single-image, no high-res splitting: a fixed 64 soft tokens.
- The SigLIP vision tower converts bit-faithfully (float CPU-parity corr ~ 1.0); single-image VQA is CPU-verified as coherent and image-grounded.
- This is the image path of SmolVLM2-500M-Video-Instruct — the package is image+text; video is not the converted path.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
