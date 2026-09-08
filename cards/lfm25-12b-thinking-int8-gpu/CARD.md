---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-thinking-int8-gpu
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
task: text-generation
---

# lfm25-12b-thinking-int8-gpu

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 lineage export + post-hoc ai-edge-quantizer (wi8fc recipe), per build_int8_gpu.sh same toolchain as ship_lfm25_gpu_variant_20260812/build_gpu_variants.sh (0.9.3-lineage GPU-safe ShortConv); quantizer recipe wi8fc |
| **Command** | `minicpm5_work/ship_lfm25_int8gpu_20260825/build_int8_gpu.sh` |
| **Quantization** | int8 dynamic (linears + embedding via wi8fc; convs float) — GSM8K 77% vs 76% for export-time conv-int8 on this finetune (build_int8_gpu.sh primary pointers) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-Thinking_int8_gpu.litertlm` | `11faec13dca0699dae1a43da30f0b5d57c64627edf6c4a6fa16b45b716db1a92` | 1187.099 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/lfm25-12b-thinking-int8-gpu__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/lfm25-12b-thinking-int8-gpu__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 860.26 | 28.83 | 270.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 75.12 | 7.03 | 3550.4 | 1893.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The published _int4/_int8 (non-gpu) siblings stall GPU delegation at 536/579 ops on Adreno (0.9.1-era ShortConv INT64 block; s4_gpu_gate/RESULTS.md rows 5/12). This _int8_gpu delegates 542/542 on the Galaxy S26 (device_runs 2026-08-25).
- Thinking bundle: the runtime needs the think opener prefilled and the bundle's declared thought channel to show reasoning; gate multi-turn behaviour at sizes where the thinking completes (family records; memory thinking-models-need-think-prefill).
- add_executor_metadata is required after post-hoc quantization (0.9.3 emits no ExecutorMetadataProto; litert-lm >=0.15 needs it) (build_int8_gpu.sh).
- One arm of the 08-25 Mac Metal sieve refused engine creation for a JP/Thinking variant while S26 OpenCL passed the same file — see ship_lfm25_int8gpu_20260825/RESULTS.md before quoting Mac Metal numbers for this family.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
