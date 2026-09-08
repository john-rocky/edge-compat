---
family: olmo2
license: apache-2.0
model_id: olmo-2-1b-instruct-q4-block32-ekv4096
source_url: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
task: text-generation
---

# olmo-2-1b-instruct-q4-block32-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | olmo2 |
| **Source** | https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (official upstream main, no fork patches) TODO |
| **Command** | `TODO` |
| **Quantization** | int4 weights — blockwise (block 32) + OCTAV optimal clipping, symmetric; embedding INT8; integer compute |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `OLMo-2-1B-Instruct_q4_block32_ekv4096.litertlm` | `669484d528d9b981ddf5057126f3b0e629e2fe793310f95959f8463eb538b811` | 888.101 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/olmo-2-1b-instruct-q4-block32-ekv4096__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/olmo-2-1b-instruct-q4-block32-ekv4096__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/olmo-2-1b-instruct-q4-block32-ekv4096__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 168.67 | 12.1 | 1300.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 210 | - | 494.66 | 22.97 | 470.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 39.96 | 2.13 | 7668.3 | 2326.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- GSM8K (n=100, greedy, 0-shot CoT): int4 63.0% vs bf16 72.0% — at 1B, 4-bit costs ~9 pt because a small model has less redundancy to absorb int4 rounding than a 3B+ (card Quality section).
- An int8 build recovers only ~2 pt (65%) for +60% size — int4 is shipped as the size/quality trade-off (card Quality section).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
