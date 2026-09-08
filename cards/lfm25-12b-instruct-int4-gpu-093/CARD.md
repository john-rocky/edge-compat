---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm25-12b-instruct-int4-gpu-093
source_url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
task: text-generation
---

# lfm25-12b-instruct-int4-gpu-093

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (via litertlm-convert minicpm5_work convert_lfm25.py + quantize_litertlm.py + add_executor_metadata.py) 0.9.3 (~/venvs/ltconv040dev: litert-converter 0.3.1, ai-edge-quantizer 0.8.0, litert-lm 0.15.0 builder; ship_lfm25_gpu_variant_20260812/RESULTS.md) |
| **Command** | `python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_instruct_fp --fp && python quantize_litertlm.py apply out_instruct_fp/model.litertlm LFM2.5-1.2B-Instruct_int4_gpu.litertlm --recipe wi4b32_wi8 --algo octav && python scripts/add_executor_metadata.py  # RESULTS.md pipeline; upstream ShortConv fix, composite lowered by converter 0.3.1` |
| **Quantization** | int4 blockwise-32 + OCTAV linears, int8 embedding, convs float (same recipe as the CPU int4 file, re-exported so it runs on the GPU — HF card) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-Instruct_int4_gpu.litertlm` | `36f7f0221bcc42c75291da1d7e3422901024a5b06b9bfa3c02d7feface04f70a` | 702.115 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm25-12b-instruct-int4-gpu-093__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-23/lfm25-12b-instruct-int4-gpu-093__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm25-12b-instruct-int4-gpu-093__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/lfm25-12b-instruct-int4-gpu-093__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 19 | - | 10.26 | 24.09 | 1890.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-23 | measured |
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 86.48 | 40.7 | 2390.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 19 | - | 80.27 | 24.74 | 280.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-23 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 1021.22 | 54.56 | 220.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| pixel-8a | gpu | pass | yes | - | 19 | - | 67.46 | 19.88 | 330.0 | - | Pixel 8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-12 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 54.58 | 9.3 | 4798.0 | 1459.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- GPU needs litert-lm >= 0.16.0 (Android OpenCL and macOS, verified by generation); iOS Metal fails at engine creation for this family, tracked upstream in LiteRT-LM#3129 — use CPU on iOS (HF card).
- On phone-class hardware the GPU's win is prefill and time-to-first-token (3-6x); decode is bandwidth-bound and roughly a wash — long prompts gain far more than long answers (HF card, measured).
- GSM8K (greedy, 0-shot CoT, n=100): int4 recipe scores 72% vs bf16 reference 79%; the int8 CPU file is the quality pick at 81% (HF card).
- The device-run table's speed columns come from the runtime's default prompt (~19-token prefill, ~48-token decode) — a latency floor, not throughput. Same artifact, same pinned litert-lm 0.16.0 binary, Galaxy S26 (SM8850), real 205-token prompt, 3 runs: prefill 1017.5 tok/s GPU (spread 1.6%) vs 92.6 CPU (spread 13%) — 12.7x the table's GPU figure; decode 41.82 GPU vs 43.34 CPU, a real 3.6% CPU win since neither range overlaps. Quote CPU prefill from repeats only (litertlm-convert gpu_s26_20260823/RESULTS.md, 2026-08-23).
- Benchmarking this artifact on GPU with --benchmark_prefill_tokens and --benchmark_decode_tokens together and no --max_num_tokens aborts engine creation: benchmark mode auto-sizes max_num_tokens to >= prefill+decode, and past ~192 total the DYNAMIC_UPDATE_SLICE shape check rejects the graph, delegation drops to 104/542 and the OpenCL shader fails to compile ('half4' vs 'float4'). Either flag alone is fine; --max_num_tokens=1024 makes it pass. Measured on litert-lm 0.16.0 / Galaxy S26 (RESULTS.md, 2026-08-23).
- 0.9.3 does not write ExecutorMetadata itself — add_executor_metadata.py is a mandatory post-step or litert-lm >= 0.15 fails at inference (RESULTS.md).
- Artifact identity is contradicted in this repo: cards lfm25-12b-int4-gpu and lfm25-12b-instruct-int4-gpu-093 both record sha256 36f7f0221bcc… (702.115 MB, same filename) while recording different conversion lineages — a 2026-07-29 convert_lfm25_patchless_092.py export with an unrecorded --algo, versus a 2026-08-12 convert_lfm25.py build on litert-torch 0.9.3 with converter 0.3.1. At most one can be true. The 07-29 artifact is no longer on disk, so which sha is wrong is UNMEASURED. Do not cite either conversion.command as settled until that export is reproduced and hashed; the published litert-community file hashes to this sha.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
