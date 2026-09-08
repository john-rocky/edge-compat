---
family: lfm2_vl
license: lfm-open-license-v1.0
model_id: lfm25-vl-450m-int4
source_url: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
task: image-text-to-text
---

# lfm25-vl-450m-int4

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
| **Command** | `python convert_lfm25_vl3b.py ../src_models/lfm25-vl-450m out_vl450m_fp2 --fp && python quantize_vl.py out_vl450m_fp2/model.litertlm LFM2.5-VL-450M_int4.litertlm` |
| **Quantization** | text int4 blockwise-32 OCTAV linears + int8 embedding/lm_head; vision tower int8 dynamic (wi4b32_wi8 + embedder wi8 pass; embedder 268 -> 68 MB) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-450M_int4.litertlm` | `45a0224180c89a25299078524e5a8dfcce131686fd0f941cdcf3507095db4797` | 387.974 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/lfm25-vl-450m-int4__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-vl-450m-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm25-vl-450m-int4__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 403.63 | 99.25 | 520.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 2699.59 | 103.02 | 90.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 296 | - | 92.12 | 22.62 | 3260.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 296 | - | 528.07 | 33.05 | 590.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- int4 is the better variant of this model on our gates — 8/8 text and 4/5 image vs the int8's 6/8 and 3/5; on Pixel 8a CPU int4 prefills slower than int8 (blockwise repacking) but decodes ~40% faster — pick by whether prompts or outputs dominate (HF card, measured).
- 64k vocab (stop ids 7/2, same as the 1.2B text family) — the 3B's 128k pbtext does not transfer; verify generation_config before reusing metadata across sizes (REPRODUCE.md).
- Fine-grained shape questions can miss on-device at this scale across all quant levels including an unquantized probe, while the torch reference answers them; conversion exonerated by experiment (vision tflite cosine 1.0000, token-identical streams, geometry probes correct). Treat as a fast small VLM for coarse visual tasks and large-text OCR; the 3B answers all fixtures (RESULTS.md; HF card wording).
- Same VLM-bundle mechanics as the siblings: transformers==5.14.1 pin; llm_model_type { lfm2 {} } with bare <image> template; externalized-embedder int8 pass; pack/unpack post-processing (single-tflite tools drop vision sections) (REPRODUCE.md).
- Zero-scale fix is a no-op on this checkpoint (0 all-zero blocks, like the 1.2B family); harmless to keep in the pipeline (RESULTS.md).
- iPhone GPU blocked (LiteRT-LM#3129) — iOS runs CPU; Android OpenCL delegates fully (543/543).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
