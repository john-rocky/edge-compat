---
family: rf-detr
license: apache-2.0
model_id: rf-detr-nano__rfdetr_graphb_fp16
source_url: https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT
task: object-detection
---

# rf-detr-nano__rfdetr_graphb_fp16

| | |
|---|---|
| **Task** | object-detection |
| **Family** | rf-detr |
| **Source** | https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python scripts/build_rfdetr_split.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `1c0a2f89217cfa41b8bfdb2f89752febfd31deba87c45278e9843e6fbb41f8fc.tflite` | `ced9a6b29f8196c6d66c9864a11f4fd2d7e23bb5ab2d0d83f49101f85321fc86` | 7.243 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/rf-detr-nano__rfdetr_graphb_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 134.482 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 235.62733693394267 | 4.797 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rf-detr-nano__rfdetr_graphb_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/rf-detr-nano__rfdetr_graphb_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 7.414 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 12.822 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 64.035 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Two-graph split: the two-stage query selection (TOPK/GATHER) runs on the host between Graph A and Graph B (zoo README L281-283).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
