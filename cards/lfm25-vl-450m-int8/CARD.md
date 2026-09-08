---
family: lfm2_vl
license: lfm-open-license-v1.0
model_id: lfm25-vl-450m-int8
source_url: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
task: image-text-to-text
---

# lfm25-vl-450m-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | lfm2_vl |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-VL-450M |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf --task image_text_to_text (via litertlm-convert lfm25vl_work/convert_lfm25_vl3b.py + quantize_vl.py; public mirror hf-to-litertlm lfm_work/convert_lfm25_vl.py + quantize_lfm25_vl.py) 0.9.3 (.venv-vl093: transformers 5.14.1, torch 2.12.1, torchvision 0.27.1) |
| **Command** | `python convert_lfm25_vl3b.py ../src_models/lfm25-vl-450m out_vl450m_int8_dir && python quantize_vl.py out_vl450m_int8_dir/model.litertlm LFM2.5-VL-450M_int8.litertlm --recipe none  # export-time dynamic int8 is the convert default (no --fp); the quantize_vl pass adds ExecutorMetadata + the externalized-embedder int8 recipe (RESULTS.md)` |
| **Quantization** | int8 dynamic (text linears + convs + embedding, vision tower) — export-time dynamic_wi8_afp32; externalized embedder int8 via quantize_vl (embedder 268 -> 68 MB) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-450M_int8.litertlm` | `f941a5f9482e31c1b7980456ac94269f2e45fffa28cd49f50c8435475c106dda` | 537.443 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/lfm25-vl-450m-int8__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-vl-450m-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm25-vl-450m-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 766.74 | 70.53 | 280.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 2345.63 | 97.57 | 100.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 296 | - | 177.25 | 15.75 | 1730.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 296 | - | 577.98 | 31.08 | 540.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- int4 is the better variant of this model on our gates — this int8 scores 6/8 text and 3/5 image vs the int4's 8/8 and 4/5; on Pixel 8a CPU int8 prefills faster than int4 but decodes ~40% slower — pick by whether prompts or outputs dominate (HF card; lfm25vl_work/RESULTS.md gate table, measured).
- Fine-grained shape questions can miss on-device at this scale across all quant levels including an unquantized probe, while the torch reference answers them; conversion exonerated by experiment (vision tflite cosine 1.0000, token-identical streams). Treat as a fast small VLM for coarse visual tasks and large-text OCR; the 3B answers all fixtures (RESULTS.md; HF card wording).
- Same VLM-bundle mechanics as the siblings: transformers==5.14.1 pin; llm_model_type { lfm2 {} } with bare <image> template; externalized-embedder int8 pass; pack/unpack post-processing (single-tflite tools drop vision sections) (REPRODUCE.md).
- iPhone GPU blocked (LiteRT-LM#3129) — iOS runs CPU; Android OpenCL delegates the text graph fully (543/543, zero rejections).
- Android v0.16.0 litert_lm_main has no image flag (--max_num_images only) — image input is API-only on Android for now, so device rows are text-graph numbers (RESULTS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
