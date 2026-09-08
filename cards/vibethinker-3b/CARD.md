---
family: qwen2
license: mit
model_id: vibethinker-3b
source_url: https://huggingface.co/litert-community/VibeThinker-3B
task: text-generation
---

# vibethinker-3b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen2 |
| **Source** | https://huggingface.co/litert-community/VibeThinker-3B |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (hf-to-litertlm reproduce_llm.sh key `vibethinker-3b`) TODO |
| **Command** | `CACHE=4096 EXTERNALIZE_EMBEDDER=1 export_simple_template.py WeiboAI/VibeThinker-3B <out> templates/chatml_simple.jinja BOCTAV4` |
| **Quantization** | int4 weights, block 32, symmetric + OCTAV optimal clipping; embeddings INT8 (externalized section) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `12a5a5089880044f8cebba1e4f5380efbfe5e55461f86cf6878d20dbb9234d40` | 1961.809 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-23/vibethinker-3b__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-17/vibethinker-3b__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/vibethinker-3b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/vibethinker-3b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 157.44 | 13.88 | 1360.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1358.96 | 93.16 | 199.1 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-23 | measured |
| pixel-8a | gpu | pass | yes | - | 17 | - | 12.6 | 7.58 | 1480.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-17 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 22.53 | 3.36 | 13618.3 | 2808.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Block 32 ONLY. This is a precision-sensitive math model: block-128 int4 (a quarter of the dequant scales) collapsed to 64% on GSM8K, −33 points, while block 32 holds at 90%. Only the block-32 build is published — and the opposite holds for general-purpose 4B reasoning models (HF card).
- Math-specialised: on general-knowledge prompts it reasons at length and can settle on a wrong answer. Measured 2026-08-17, its CPU and GPU legs are wrong in the same shape with different hallucinated content, which is how the behaviour is attributable to the model rather than the backend (CATALOG_REGATE.md).
- Measured on an 8 GB phone 2026-08-17: fully delegated on Mali OpenCL — 1603/1603 prefill nodes, 1452/1452 decode, zero rejections. Neither leg printed BenchmarkInfo because the model generated to the token cap, so speed figures for this model need a prompt it will actually finish.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
