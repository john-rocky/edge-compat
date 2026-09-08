---
family: smollm3
license: apache-2.0
model_id: smollm3-3b
source_url: https://huggingface.co/litert-community/SmolLM3-3B
task: text-generation
---

# smollm3-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | smollm3 |
| **Source** | https://huggingface.co/litert-community/SmolLM3-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (hf-to-litertlm reproduce_llm.sh key `smollm3-3b`) TODO |
| **Command** | `CACHE=4096 EXTERNALIZE_EMBEDDER=1 export_simple_template.py HuggingFaceTB/SmolLM3-3B <out> templates/smollm3_think.jinja BOCTAV4` |
| **Quantization** | int4 weights, blockwise-32 + OCTAV optimal clipping, symmetric; embedding INT8 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `SmolLM3-3B_q4_block32_ekv4096.litertlm` | `38d7bf55e243f5a95d8b16c21b1f7ef7debe6ce6bb2ddb66eb54f4bc2be8e6eb` | 1909.502 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-22/smollm3-3b__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-17/smollm3-3b__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/smollm3-3b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/smollm3-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/smollm3-3b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 442 | - | 147.93 | 9.81 | 3090.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 267.97 | 14.84 | 820.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-22 | measured |
| pixel-8a | gpu | pass | yes | - | 17 | - | 11.89 | 7.69 | 1560.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-17 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 18.32 | 2.46 | 16328.9 | 3194.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Thinking model: the bundle carries the SmolLM3 template with its /think and /no_think controls, not plain ChatML (HF card).
- Gallery import: only v1.0.16+ accepts .litertlm directly from Hugging Face; older 1.0.x builds (package com.google.aiedge.gallery) do not (HF card).
- Measured on an 8 GB phone 2026-08-17: fully delegated on Mali OpenCL — 1476/1476 prefill nodes, 1308/1308 decode, zero rejections — and answers correctly. This refutes the earlier standing assumption that a ~2B-and-up dense model cannot use the Pixel 8a GPU at all (CATALOG_REGATE.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
