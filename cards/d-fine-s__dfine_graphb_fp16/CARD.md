---
family: d-fine
license: apache-2.0
model_id: d-fine-s__dfine_graphb_fp16
source_url: https://huggingface.co/litert-community/D-FINE-S-LiteRT
task: object-detection
---

# d-fine-s__dfine_graphb_fp16

| | |
|---|---|
| **Task** | object-detection |
| **Family** | d-fine |
| **Source** | https://huggingface.co/litert-community/D-FINE-S-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python scripts/build_dfine_split.py && python scripts/build_dfine_fix3.py && python scripts/pack_assets.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `08ac69d8aab391c36d1918c3bbce48ce3aea646f4a88457552904482952db3c7.tflite` | `628eb9b3b8be096b57f626c7a0e01403781c3dd37466bea7bcdaf3e63076e45d` | 8.41 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/d-fine-s__dfine_graphb_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 432.95 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 43274.63062780912 | 25.665 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/d-fine-s__dfine_graphb_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/d-fine-s__dfine_graphb_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 134.036 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 172.363 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 1018.76 | - | - | - | 426.64 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Graph B consumes the host-computed top-300 selection (memory, target, ref); sigmoid + threshold + cxcywh->xyxy + light NMS are host-side (HF card 'How it runs').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
