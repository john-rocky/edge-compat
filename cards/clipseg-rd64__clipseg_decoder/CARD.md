---
family: clipseg
license: apache-2.0
model_id: clipseg-rd64__clipseg_decoder
source_url: https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT
task: image-segmentation
---

# clipseg-rd64__clipseg_decoder

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | clipseg |
| **Source** | https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `~/clipconv/bin/python build_clipseg.py` |
| **Quantization** | none — fp32 (artifact tensors: 271 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `d9683859c2314380e376e91ae659f6d8c4ea423008ae02a6db62045cfcd2fe18.tflite` | `2f06bf9438b2dd7d236a80296f4c9fd41d6ceaac38763d54ae9988e0fc7bb719` | 4.337 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/clipseg-rd64__clipseg_decoder.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 21.015 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.18152610441767067 | 4.105 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/clipseg-rd64__clipseg_decoder__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-25/clipseg-rd64__clipseg_decoder__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 3.252 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-25 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 1.67 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-25 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 19.04 | - | - | - | 135.48 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The decoder must run on CPU: its 4-head/head_dim-16 attention fp16-miscomputes on the Mali GPU delegate (the 12-head/head_dim-64 vision encoder survives at 0.998) (zoo README L1409-1411).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
