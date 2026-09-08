---
family: lfm2_vl
license: lfm-open-license-v1.0
model_id: lfm25-vl-3b-int4
source_url: https://huggingface.co/LiquidAI/LFM2.5-VL-3B
task: image-text-to-text
---

# lfm25-vl-3b-int4

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | lfm2_vl |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-VL-3B |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf --task image_text_to_text (via litertlm-convert lfm25vl_work/convert_lfm25_vl3b.py + quantize_vl.py; public mirror hf-to-litertlm lfm_work/convert_lfm25_vl.py + quantize_lfm25_vl.py) 0.9.3 (.venv-vl093: transformers 5.14.1, torch 2.12.1, torchvision 0.27.1) |
| **Command** | `python convert_lfm25_vl3b.py ../src_models/lfm25-vl-3b out_vl3b_fp --fp && python quantize_vl.py out_vl3b_fp/model.litertlm LFM2.5-VL-3B_int4.litertlm` |
| **Quantization** | text int4 blockwise-32 OCTAV linears + int8 embedding/lm_head; vision tower int8 dynamic (wi4b32_wi8 + embedder wi8 pass) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-VL-3B_int4.litertlm` | `ebb563f3587feb0cfa6867ff53fba556043ee459410a6a0276f3245e20ba0a59` | 2243.068 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/lfm25-vl-3b-int4__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-vl-3b-int4__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 471.56 | 27.04 | 470.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 292 | - | 14.12 | 5.77 | 20850.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 292 | - | 91.63 | 10.79 | 3280.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- Pin transformers==5.14.1 for the export: 5.15.0 renamed Lfm2ShortConv.L_cache -> conv_kernel_size and litert-torch 0.9.3's subclass still reads the old attribute (AttributeError at model load). torchvision must stay 0.27.x to keep torch under litert-torch's <2.13 pin (REPRODUCE.md).
- The metadata pbtext needs llm_model_type { lfm2 {} } — the empty message is correct (runtime Lfm2DataProcessor proto defaults equal this family's processor config). The chat template must render a bare <image> per image part and never boi/eoi: the runtime splits on the marker and inserts <|image_start|> + pixels + <|image_end|> itself (REPRODUCE.md).
- The externalized VLM embedder dodges every text quantization recipe: an --fp text export leaves it float32 (1 GB at the 128k vocab) and a post-hoc recipe applied to prefill_decode never reaches it — quantize it int8 as a separate pass (1049 -> 264 MB; REPRODUCE.md / quantize_vl.py).
- VLM bundles have >1 tflite section: single-tflite post-processing tools (quantize_litertlm.py, fix_zero_block_scales.py main()) rebuild via litert-lm-builder with one tflite and silently DROP the vision sections — post-process through litert-lm pack/unpack instead (RESULTS.md trap).
- The zero-scale int4 wall is inherited from the dense LFM2.5 lineage: this 3B checkpoint carries 746,432 all-zero 32-blocks across 26 tensors, so the zero-scale fix is mandatory or XNNPACK refuses the file (REPRODUCE.md; same signature as LFM2.5-2.6B).
- iPhone GPU is blocked for the ShortConv family at engine creation (upstream LiteRT-LM#3129, re-verified on 0.16.0) — iOS runs CPU; Android OpenCL and macOS GPU delegate fully (937/937 nodes, HF card).
- On a nearly-full Android device the GPU path's multi-GB compile-cache write can fail mid-initialization and masquerade as a kernel failure — run with --disable_cache=true (bare --disable_cache does not take) or free storage (RESULTS.md trap, hit on Pixel 8a).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
