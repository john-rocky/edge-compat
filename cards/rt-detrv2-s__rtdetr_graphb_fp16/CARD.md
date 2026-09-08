---
family: rt-detr
license: apache-2.0
model_id: rt-detrv2-s__rtdetr_graphb_fp16
source_url: https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT
task: object-detection
---

# rt-detrv2-s__rtdetr_graphb_fp16

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
| `c6f2cf18f74da0f002d3aef97473538e6dc41d233cf47c9ad47afc5dc16c54e5.tflite` | `fef385321b21a637652b06e70bb80a7d41768e9e2332cebaa86a24af5c9190b8` | 7.305 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/rt-detrv2-s__rtdetr_graphb_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 634.398 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.000570699019493527 | 27.723 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rt-detrv2-s__rtdetr_graphb_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/rt-detrv2-s__rtdetr_graphb_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 166.556 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 132.551 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 1199.458 | - | - | - | 540.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Graph B consumes the host-computed top-300 selection; sigmoid + threshold + cxcywh->xyxy + light NMS are host-side (zoo README L314-315).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
