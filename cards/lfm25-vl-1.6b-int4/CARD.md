---
family: lfm2_vl
license: lfm-open-license-v1.0
model_id: lfm25-vl-1.6b-int4
source_url: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
task: image-text-to-text
---

# lfm25-vl-1.6b-int4

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
| **Command** | `python convert_lfm25_vl3b.py ../src_models/lfm25-vl-16b out_vl16b_fp --fp && python quantize_vl.py out_vl16b_fp/model.litertlm LFM2.5-VL-1.6B_int4.litertlm` |
| **Quantization** | text int4 blockwise-32 OCTAV linears + int8 embedding/lm_head; vision tower int8 dynamic (wi4b32_wi8 + embedder wi8 pass; embedder 537 -> 135 MB) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-1.6B_int4.litertlm` | `44816005b42bef8887bc3dc40b7d6483ea4dd8569fb1591f00faa33ef3043e8d` | 1238.005 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/lfm25-vl-1.6b-int4__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-vl-1.6b-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm25-vl-1.6b-int4__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 127.27 | 50.85 | 1630.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 1020.13 | 54.1 | 220.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 296 | - | 36.92 | 13.64 | 8090.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 296 | - | 201.04 | 22.08 | 1520.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- Vocab split inside one family: the 1.6B and 450M use the old 64k vocab (stop ids 7/2, same as the 1.2B text family), the 3B the 128k one (124900/124895) — verify generation_config before reusing a metadata pbtext across sizes (REPRODUCE.md).
- Text is the family's best on our gates (8/8 on all four int4/int8 x cpu/gpu configs), but fine-grained shape/counting image fixtures deterministically miss on-device across ALL quant levels including an unquantized probe, while the torch reference answers them — with conversion exonerated by experiment (vision+projector tflite vs torch cosine 1.0000; token-identical prompt streams; on-device geometry probes correct; chunk-boundary alignment probe negative). Engine-internal residue; the 3B is unaffected. Card as coarse-understanding/OCR/localization-strong, defer fine shape/counting to the 3B (RESULTS.md elimination log; REPRODUCE.md warning block).
- Same VLM-bundle mechanics as the 3B: transformers==5.14.1 pin; llm_model_type { lfm2 {} } with bare <image> template; externalized-embedder int8 pass; pack/unpack post-processing (single-tflite tools drop vision sections) (REPRODUCE.md).
- int4 zero-scale fix found 0 all-zero blocks on this checkpoint (like the 450M and the 1.2B family) — only the 3B/2.6B dense lineage carries the 746k-block signature; run the fix regardless, it is a no-op when clean (RESULTS.md).
- iPhone GPU blocked (LiteRT-LM#3129) — iOS runs CPU; Android OpenCL delegates fully (543/543) and macOS GPU is measured on the HF card.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
