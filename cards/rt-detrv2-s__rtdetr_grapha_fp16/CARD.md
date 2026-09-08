---
family: rt-detr
license: apache-2.0
model_id: rt-detrv2-s__rtdetr_grapha_fp16
source_url: https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT
task: object-detection
---

# rt-detrv2-s__rtdetr_grapha_fp16

| | |
|---|---|
| **Task** | object-detection |
| **Family** | rt-detr |
| **Source** | https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python scripts/build_rtdetr_split.py && python scripts/build_rtdetr_fix3.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `d363d83a67717b6d9a86429e8067a8452925f4e7fe3767ff629577cd62466159.tflite` | `b604c6a6907549cf173133f39d503da96a0a737eab8f62e716419075a3323034` | 32.198 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/rt-detrv2-s__rtdetr_grapha_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 3315.158 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 74.50580596923828 | 52.588 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rt-detrv2-s__rtdetr_grapha_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/rt-detrv2-s__rtdetr_grapha_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 23.066 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 21.128 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 523.099 | - | - | - | 227.3 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Graph A emits only the two clean leaves (enc_class + memory_raw) and the per-token tail runs on the host over the 300 selected tokens — the workaround for a Mali bug that silently corrupts a fanned-out 3D token tensor [1,N,256] (zoo README L302-305).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
