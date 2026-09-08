---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-jp-int4-gpu-093
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP
task: text-generation
---

# lfm25-12b-jp-int4-gpu-093

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via litertlm-convert minicpm5_work convert_lfm25.py + quantize_litertlm.py + add_executor_metadata.py) 0.9.3 (~/venvs/ltconv040dev: litert-converter 0.3.1, ai-edge-quantizer 0.8.0, litert-lm 0.15.0 builder; ship_lfm25_gpu_variant_20260812/RESULTS.md) |
| **Command** | `python convert_lfm25.py LiquidAI/LFM2.5-1.2B-JP out_jp_fp --fp && python quantize_litertlm.py apply out_jp_fp/model.litertlm LFM2.5-1.2B-JP_int4_gpu.litertlm --recipe wi4b32_wi8 --algo octav && python scripts/add_executor_metadata.py  # RESULTS.md pipeline; upstream ShortConv fix, composite lowered by converter 0.3.1` |
| **Quantization** | int4 blockwise-32 + OCTAV linears, int8 embedding, convs float (same recipe as the CPU int4 file, re-exported so it runs on the GPU — HF card) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-JP_int4_gpu.litertlm` | `2c826a9cdeefc614c9b7de1d1c5565c9a64f704e5f3291ad2cd5d7da396790d5` | 702.115 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm25-12b-jp-int4-gpu-093__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/lfm25-12b-jp-int4-gpu-093__galaxy-s26.json`, `data/device_runs/0.16.0/2026-08-25/lfm25-12b-jp-int4-gpu-093__iphone-17-pro.json`, `data/device_runs/0.16.1/2026-09-01/lfm25-12b-jp-int4-gpu-093__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 918.98 | 54.63 | 240.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| iphone-17-pro | gpu | pass | yes | - | - | - | - | 70.02 | 91.0 | 556.0 | iphone-17-pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-25 | measured |
| pixel-8a | gpu | pass | yes | - | 19 | - | 68.05 | 20.48 | 330.0 | - | Pixel 8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-12 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 54.66 | 9.24 | 4791.6 | 1454.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- GPU needs litert-lm >= 0.16.0 (Android OpenCL and macOS, verified by generation); iOS Metal fails at engine creation for this family (LiteRT-LM#3129) — use CPU on iOS (HF card).
- On phone-class hardware the GPU's win is prefill and TTFT (3-6x); decode is bandwidth-bound and roughly a wash (HF card, measured).
- This tune's convs must stay float in int8 recipes: quantizing the JP tune's convs costs ~9pt GSM8K, unlike the Instruct sibling where conv-int8 is free; the int4 recipe here keeps convs float anyway (HF card).
- English GSM8K undersells a Japanese-optimized tune (int4 55% vs bf16 63%) — reported for quantization-fidelity transparency; Japanese conversation quality was verified by inspection (HF card).
- 0.9.3 does not write ExecutorMetadata itself — add_executor_metadata.py is a mandatory post-step or litert-lm >= 0.15 fails at inference (RESULTS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
