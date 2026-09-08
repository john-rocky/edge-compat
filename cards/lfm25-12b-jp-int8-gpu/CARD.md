---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-jp-int8-gpu
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP
task: text-generation
---

# lfm25-12b-jp-int8-gpu

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 lineage export + post-hoc ai-edge-quantizer (wi8fc recipe), per build_int8_gpu.sh same toolchain as ship_lfm25_gpu_variant_20260812/build_gpu_variants.sh (0.9.3-lineage GPU-safe ShortConv); quantizer recipe wi8fc |
| **Command** | `minicpm5_work/ship_lfm25_int8gpu_20260825/build_int8_gpu.sh` |
| **Quantization** | int8 dynamic (linears + embedding via wi8fc; convs float) — same weights class as the published _int8 card describes |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-JP_int8_gpu.litertlm` | `1316b5381a45f01b1a06da6ebe1a99bb55b430653e6a7ee19d50a6d0ad01d869` | 1187.099 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/lfm25-12b-jp-int8-gpu__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/lfm25-12b-jp-int8-gpu__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 940.41 | 41.86 | 240.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 75.33 | 7.01 | 3541.0 | 1893.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The published _int4/_int8 (non-gpu) siblings stall GPU delegation at 536/579 ops on Adreno — litert-torch 0.9.1-era Lfm2ShortConv_conv leaves INT64 ADD/CAST/SUM, GATHER_ND, const-input GREATER_EQUAL/LESS_EQUAL in the graph (s4_gpu_gate/RESULTS.md rows 4/11). This _int8_gpu carries the 0.9.3-lineage ShortConv and delegates 542/542 on the Galaxy S26 (device_runs 2026-08-25).
- Quantizer recipe matters per-finetune: export-time conv-int8 costs JP 9 GSM8K points vs wi8fc (56% vs 65%), while Instruct gains 2 points from export-time — the A/B is per-model, not per-family (build_int8_gpu.sh primary pointers; memory lfm25-12b-shipped).
- add_executor_metadata is required: litert-torch 0.9.3 does not emit ExecutorMetadataProto and quantize_litertlm.py rebuilds the bundle; litert-lm >=0.15 needs it for the hybrid conv/attention state buffers (build_int8_gpu.sh).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
