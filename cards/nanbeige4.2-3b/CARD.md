---
family: nanbeige
license: apache-2.0
model_id: nanbeige4.2-3b
source_url: https://huggingface.co/Nanbeige/Nanbeige4.2-3B
task: text-generation
---

# nanbeige4.2-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nanbeige |
| **Source** | https://huggingface.co/Nanbeige/Nanbeige4.2-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm hf-to-litertlm a59a492 (2026-08-01 public mirror of the conversion-session state) |
| **Command** | `bash scripts/reproduce_llm.sh nanbeige4.2-3b` |
| **Quantization** | int4 blockwise-32 + OCTAV clipping, symmetric; embedding int8 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `4b4adfe6fd794451116bca61db58f0e2fb9fc3542da8d04d11ba4635693f4533` | 2459.791 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/nanbeige4.2-3b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/nanbeige4.2-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/nanbeige4.2-3b__raspberry-pi-5.json`, `data/device_runs/0.17.0/2026-09-06/nanbeige4.2-3b__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 225 | - | 20.92 | 4.1 | 11000.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | load_failed | yes | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 5.73 | 1.06 | 45855.5 | 4311.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Sampling is required: the model collapses under greedy decoding — use the official recipe (temperature 0.6, top_k 20, top_p 0.95, per its generation_config.json); do not run at temperature 0.
- CPU backend only: the current GPU delegate produces incorrect output on this 44-layer unrolled graph.
- Looped transformer (22 layers x num_loops=2, unrolled at export): ~2x the per-token compute of a normal 3B — expect roughly 8B-class decode rates; 44 KV-cache slots, one per (loop, layer).
- transformers 5.x zeroes the modeling's init-computed rotary inv_freq buffer — rope must be recomputed from config at export.
- The chat template auto-opens the model's <think> block (official template behaviour); the visible answer follows </think>.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
