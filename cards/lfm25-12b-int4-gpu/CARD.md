---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-int4-gpu
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
task: text-generation
---

# lfm25-12b-int4-gpu

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via litertlm-convert minicpm5_work/convert_lfm25_patchless_092.py + quantize_minicpm5.py + litert-lm-builder) 0.9.2 (stock release, per convert_lfm25_patchless_092.py docstring + .venv-092 dist-info) |
| **Command** | `python minicpm5_work/convert_lfm25_patchless_092.py LiquidAI/LFM2.5-1.2B-Instruct <outdir> && python minicpm5_work/quantize_minicpm5.py apply <in>.litertlm <out>.litertlm --recipe wi4b32_wi8 --algo <unrecorded: minmax or octav; not recoverable, owner does not recall> && python scripts/add_executor_metadata.py <out>.litertlm   # export 2026-07-29; metadata retrofit 2026-08-11` |
| **Quantization** | int4 blockwise-32 weights (fp16 per-block scales) + int8 embedding/lm_head; convs float — artifact tensor census; algorithm unrecorded |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-Instruct_int4_gpu.litertlm` | `36f7f0221bcc42c75291da1d7e3422901024a5b06b9bfa3c02d7feface04f70a` | 702.115 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm25-12b-int4-gpu__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-26/lfm25-12b-int4-gpu__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-27/lfm25-12b-int4-gpu__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iphone-17-pro | gpu | pass | - | - | - | - | 496.37 | 69.53 | 84.0 | 412.175 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-26 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 1505.16 | 325.76 | 27.0 | 711.814 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 · iOS Version 27.0 (Build 26A5416b) | 2026-08-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 4662.61 | 346.72 | 57.8 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |

## Pitfalls

- The raw 0.9.2 export lacks the ExecutorMetadata section and dies at executor.cc:708 on litert-lm >=0.15; this artifact is the execmeta-retrofitted copy (2026-08-11) of the 2026-07-29 export (ship_lfm25_gpu_variant_20260812/RESULTS.md; file naming *_execmeta).
- iPhone Metal fails engine creation on this file family (v0.14: DUS updated_slice>operand at delegate-resolved shapes; the static graph is clean across all 530 subgraphs — delegate-side) while Mac WebGPU compiles and passes the same file at maxtok 1024 AND 2048 (DEVICE_GATE_iphone17pro_20260729.md).
- Superseded on HF: litert-community's LFM2.5-1.2B-Instruct_int4_gpu.litertlm is a 2026-08-12 re-export (litert-torch 0.9.3 + converter 0.3.1) with different bytes; this card's sha256 identifies the 07-29 0.9.2 canary file that the recorded device runs measured (ship_lfm25_gpu_variant_20260812/RESULTS.md).
- Artifact identity is contradicted in this repo: cards lfm25-12b-int4-gpu and lfm25-12b-instruct-int4-gpu-093 both record sha256 36f7f0221bcc… (702.115 MB, same filename) while recording different conversion lineages — a 2026-07-29 convert_lfm25_patchless_092.py export with an unrecorded --algo, versus a 2026-08-12 convert_lfm25.py build on litert-torch 0.9.3 with converter 0.3.1. At most one can be true. The 07-29 artifact is no longer on disk, so which sha is wrong is UNMEASURED. Do not cite either conversion.command as settled until that export is reproduced and hashed; the published litert-community file hashes to this sha.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
