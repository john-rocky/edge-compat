---
family: lfm2_vl
license: lfm-open-license-v1.0
model_id: lfm25-vl-1.6b-int8
source_url: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
task: image-text-to-text
---

# lfm25-vl-1.6b-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | lfm2_vl |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf --task image_text_to_text (via litertlm-convert lfm25vl_work/convert_lfm25_vl3b.py + quantize_vl.py; public mirror hf-to-litertlm lfm_work/convert_lfm25_vl.py + quantize_lfm25_vl.py) 0.9.3 (.venv-vl093: transformers 5.14.1, torch 2.12.1, torchvision 0.27.1) |
| **Command** | `python convert_lfm25_vl3b.py ../src_models/lfm25-vl-1.6b out_vl16b_int8 && python quantize_vl.py out_vl16b_int8/model.litertlm LFM2.5-VL-1.6B_int8.litertlm --recipe none  # export-time dynamic int8 is the convert default (no --fp); the quantize_vl pass adds ExecutorMetadata + the externalized-embedder int8 recipe (RESULTS.md)` |
| **Quantization** | int8 dynamic (text linears + convs + embedding, vision tower) — export-time dynamic_wi8_afp32; externalized embedder int8 via quantize_vl |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-1.6B_int8.litertlm` | `d6a254aa3edefab51684918e7c717d096c60e495c085fcc35fd845fb8c80c8bd` | 1725.193 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/lfm25-vl-1.6b-int8__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-vl-1.6b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm25-vl-1.6b-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 525.76 | 41.66 | 410.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 1518.86 | 42.27 | 160.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 296 | - | 60.42 | 8.38 | 5020.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 296 | - | 403.31 | 15.91 | 800.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- int4 vs int8 trade on this model: int8 prefills faster on CPU, int4 decodes markedly faster (blockwise-int4 repacking cost sits in prefill) — pick by whether prompts or outputs dominate; text graph delegates fully on Android OpenCL (543/543, zero rejections) (HF card, measured).
- Same VLM-bundle mechanics as the siblings: transformers==5.14.1 pin; llm_model_type { lfm2 {} } with bare <image> template; externalized-embedder int8 pass; pack/unpack post-processing (single-tflite tools drop vision sections) (REPRODUCE.md).
- iPhone GPU blocked (LiteRT-LM#3129) — iOS runs CPU; use litert-lm >= 0.16.0 for Android OpenCL / macOS GPU (HF card).
- Android v0.16.0 litert_lm_main has no image flag (--max_num_images only) — image input is API-only on Android for now, so device rows are text-graph numbers (RESULTS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
