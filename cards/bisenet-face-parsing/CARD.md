---
family: bisenet
license: mit
model_id: bisenet-face-parsing
source_url: https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT
task: image-segmentation
---

# bisenet-face-parsing

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | bisenet |
| **Source** | https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python faceparsing/scripts/build_faceparsing.py` |
| **Quantization** | none — fp32 (artifact tensors: 142 float32, no float16/int8) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `f5a806696ecbdb715b4ec22c27eaab348874efa00bc1df2af6a3db52963aa408.tflite` | `23923022a8f4149fe1f5df5426a18f3a202dd605454539ed33c5d76f3035e907` | 50.201 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/bisenet-face-parsing.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 442.142 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.6973950795947902 | 79.592 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/bisenet-face-parsing__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 249.388 | - | - | - | 218.56 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
