---
family: paddleocr-vl
license: apache-2.0
model_id: paddleocr-vl-1.6
source_url: https://huggingface.co/litert-community/PaddleOCR-VL-1.6
task: image-text-to-text
---

# paddleocr-vl-1.6

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | paddleocr-vl |
| **Source** | https://huggingface.co/litert-community/PaddleOCR-VL-1.6 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (LiteRT-LM fast_vlm bundle: static-NaViT VISION_ENCODER [1,560,560,3]->[1,1600,1152] + VISION_ADAPTER [1,1600,1152]->[1,400,1024] + single-token EMBEDDER + PREFILL_DECODE; the ERNIE-4.5-0.3B decoder re-hosted as a standalone LlamaForCausalLM, cache 4096) (HF card Conversion notes) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (not recorded in the HF card)` |
| **Quantization** | decoder shipped fp16 (int4 and integer-compute int8 measurably corrupt transcription on this 0.36B decoder, ~460 MB spent on exactness); vision tower/adapter recipe not stated (HF card Quality + Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `PaddleOCR-VL-1.6.litertlm` | `4dd0a268b1849a95e949f546de201c9957bf5acc98494c150bd02165c171cc76` | 1325.899 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/paddleocr-vl-1.6__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/paddleocr-vl-1.6__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/paddleocr-vl-1.6__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | run_failed | no | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 204 | - | 1238.83 | 35.59 | 190.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | run_failed | no | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Task-prompted model: you select what it does with the text prompt (OCR:, Table Recognition:, …); page OCR on a synthetic report page transcribed perfectly in 22 s, table recognition 17 s (HF card Quality).
- Static NaViT rewrite: the encoder is dynamic-resolution (packed patches, interpolated position embeddings, 2-D rotary over h/w ids) and does not torch.export; the LM calls it with full attention (window_size=-1), so a static whole-image graph is used; the projector's 2x2 spatial merge is done GPU-safe with 4 strided slices + concat (all tensors <= 4D) instead of the literal 6-D rearrange (HF card Conversion notes).
- Decoder: bit-exact fp32 logits vs the original as a standalone Llama-layout model; shipped fp16 weights are teacher-forced-parity corr 1.0000, top-1 10/10 (HF card Quality).
- M-RoPE note: the base decoder uses Qwen2-VL-style 3-D M-RoPE while the fast_vlm contract supplies plain sequential positions — identical for text tokens, and an A/B eager test (true M-RoPE vs 1-D) showed no quality loss on OCR/table tasks (HF card).
- Tokenizer: the base SP model lacks the 1,019 added tokens (<|IMAGE_START|>, <|LOC_0|>…<|LOC_1000|>, <fcel>/<nl> table tokens) — appended as USER_DEFINED pieces at their exact ids (vocab padded to 103,424) so table/spotting output detokenizes on device; prompt template baked in: <|begin_of_sentence|>User: <image>PROMPT\nAssistant:\n, stop </s> (HF card Conversion notes).
- On the Galaxy S26 and Pixel 8a CPU generation gates (S7 backfill, 2026-09-05) the text-only gate prompt produced a repeated-character flood (degenerate_output rows in data/device_runs/) — an OCR-prompt model asked a plain question; not a card-quality claim either way (device runs).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
