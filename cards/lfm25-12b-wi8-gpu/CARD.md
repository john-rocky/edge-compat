---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-wi8-gpu
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
task: text-generation
---

# lfm25-12b-wi8-gpu

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via hf-to-litertlm lfm_work/convert_lfm25.py) main checkout 1b96312 (post-0.9.2; includes DUS-folder fix 1ed2caaaf; per DEVICE_GATE 2026-08-05) |
| **Command** | `python hf-to-litertlm/lfm_work/convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct ship_lfm25_gpu_0150/out_wi8` |
| **Quantization** | int8 dynamic weights / fp32 activations (litert-torch recipe dynamic_wi8_afp32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `dcf7e0e17f7a4a829440151ab38f6b9f649b3a8736b559b73e3944f08a59ce94` | 1189.502 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm25-12b-wi8-gpu__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 4836.61 | 272.0 | 56.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |

## Pitfalls

- Exported on litert-torch main 1b96312 specifically to pick up the DUS-folder fix (1ed2caaaf, post-0.9.2) and the INT64-free ShortConv lowering; 0.9.1-lineage exports leave INT64 ADD/CAST/SUM + GATHER_ND in Lfm2ShortConv_conv and stall GPU delegation at 536/579 ops (DEVICE_GATE 2026-08-05; ship_lfm25_gpu_variant_20260812/RESULTS.md §1).
- litert-torch main writes the ExecutorMetadata section natively — release 0.9.3 does not; litert-lm >=0.15 requires it for the hybrid conv/attention state buffers (regate_softmax_20260811/RESULTS.md 手順 4).
- iOS Metal still fails engine creation on this file (ml_drift Metal codegen: implicit half4*float4 conversion, metal/common.mm) while the SAME file fully delegates and passes on Mac WebGPU at 4780/271.6 tok/s — runtime-side wall, re-test on each litert-lm release (DEVICE_GATE 2026-08-05).
- odml.softmax composites are stripped at export in this build (converter 0.3.0-era did not lower them; litert-converter >=0.3.1 lowers natively — see the lfm25-12b-wi8-composite variant that verifies this) (DEVICE_GATE 2026-08-05; regate_softmax_20260811/RESULTS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
