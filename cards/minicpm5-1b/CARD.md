---
family: minicpm5
license: apache-2.0
model_id: minicpm5-1b
source_url: https://huggingface.co/litert-community/MiniCPM5-1B
task: text-generation
---

# minicpm5-1b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | minicpm5 |
| **Source** | https://huggingface.co/litert-community/MiniCPM5-1B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | official LiteRT-LM release of MiniCPM5-1B (README 'Available Models'); not converted in this lane n/a (official artifact) |
| **Command** | `n/a (official artifact)` |
| **Quantization** | mixed INT4-block32 (linear) / INT8 (embed and lm_head) quantization (wi4b32_wi8) with FP32 activations (afp32) (README Available Models) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `minicpm_wi4b32_wi8_afp32.litertlm` | `43f41a837c58408192d703a7cf7bb27a3f8b4f8cc1d34c7f79e54f058f966ccc` | 755.906 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.0/2026-09-06/minicpm5-1b__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |

## Pitfalls

- The repo also ships minicpm_wi4b32_wi8_afp32_gpu_opt.litertlm (same recipe, optimized for GPU execution) and MiniCPM5-1B_dynamic_wi8_afp32.litertlm (dynamic weight-only INT8, static prefill memory allocation) — separate artifacts, not this card (README Available Models).
- Quantization benchmark in the README compares the FP (bf16) baseline and the W4 model with thinking mode turned off to reduce context usage (README Quantization Benchmark).
- The owner's MiniCPM5-2B conversion (cards minicpm5-2b-int4/-int8) follows this artifact's packaging (verbatim chat_template.jinja on the jinja path) — see minicpm52b_work/FINDINGS.md premises.
- License Apache-2.0, consistent with upstream openbmb/MiniCPM5-1B (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
