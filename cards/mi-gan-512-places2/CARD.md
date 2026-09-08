---
family: mi-gan
license: mit
model_id: mi-gan-512-places2
source_url: https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT
task: image-to-image
---

# mi-gan-512-places2

| | |
|---|---|
| **Task** | image-to-image |
| **Family** | mi-gan |
| **Source** | https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python migan/scripts/build_migan.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `0416d52e68c7eb9aaf5d3bf670865acc935060195b99513d9240945f0c6b98d1.tflite` | `ef53f8dca69e5ce29629441128322c4d9a0a527b2a043703e0ab497f2463c25f` | 15.557 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/mi-gan-512-places2.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 3047.593 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 0.697452229299363 | 22.02 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/mi-gan-512-places2__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/mi-gan-512-places2__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 26.645 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 30.851 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 1788.29 | - | - | - | 392.61 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Input is concat(mask-0.5, rgb*mask) with rgb in [-1,1], mask 1=keep / 0=erase; composite back as rgb*mask + out*(1-mask) (zoo README L850-852).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
