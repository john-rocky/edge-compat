---
family: yunet
license: apache-2.0
model_id: gfpgan-v1.4__yunet_fp16
source_url: https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT
task: image-to-image
---

# gfpgan-v1.4__yunet_fp16

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | yunet |
| **Source** | https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (zoo yunet/scripts/build_yunet.py; fp16 via ai_edge_quantizer float_casting) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python yunet/scripts/build_yunet.py   # defaults YN_SIZE=640 YN_VAR=yunet_n; output shared byte-identical with YuNet-Face-LiteRT` |
| **Quantization** | fp16 (from artifact file name) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `yunet_fp16.tflite` | `ced5f52bef6e76ad4a66d1055b2b404336ceafbae8eeac8fed6aa9c7b2e4d776` | 0.244 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/gfpgan-v1.4__yunet_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/gfpgan-v1.4__yunet_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 3.277 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 1.828 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 33.537 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Used for face detection + 5-landmark FFHQ alignment ahead of GFPGAN restoration in the shipped pipeline (HF card 'Pipeline'; gfpgan/README.md L40-46).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
