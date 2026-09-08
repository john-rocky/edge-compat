---
family: 6drepnet
license: mit
model_id: 6drepnet-headpose
source_url: https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT
task: image-classification
---

# 6drepnet-headpose

| | |
|---|---|
| **Task** | image-classification |
| **Family** | 6drepnet |
| **Source** | https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python sixdrepnet/scripts/build_6drepnet.py` |
| **Quantization** | none — fp32 (artifact tensors: 95 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `df99d6912c5cfec3cfd2c92b8509578d1c84f628a5f74cc9b8301862d5d598dd.tflite` | `44cc9679a18242fac31c9ad369465caad93a34c721eed3fe0ff003f712769dee` | 150.04 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/6drepnet-headpose.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 240.73 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 1.5780176979400015e-05 | 17.325 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/6drepnet-headpose__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 140.556 | - | - | - | 324.09 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Use the deploy weights (fused rbr_reparam) — the conversion is of deploy-mode RepVGG with plain convs (zoo README L632).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
