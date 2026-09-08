---
family: mimi
license: cc-by-4.0
model_id: mimi__mimi_dec_tx_fp16
source_url: https://huggingface.co/litert-community/Mimi
task: neural-codec
---

# mimi__mimi_dec_tx_fp16

| | |
|---|---|
| **Task** | neural-codec |
| **Family** | mimi |
| **Source** | https://huggingface.co/litert-community/Mimi |
| **License** | cc-by-4.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python mimi/scripts/build_hybrid_graphs.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `f2f0421f2ac6d3fb17892450bf765ca57cfaea2b58b5af79955997d41c2f1db4.tflite` | `35556c72093aaea15aab2dce41946398d216fb1fdf0803d225a5719a64c740f8` | 48.196 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/mimi__mimi_dec_tx_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 40.158 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 7026.186693822132 | 4.683 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/mimi__mimi_dec_tx_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/mimi__mimi_dec_tx_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 4.097 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 2.088 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 35.364 | - | - | - | 185.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The decoder transformer runs on CompiledModel CPU: its residual stream reaches |x|=27 and Mali fp16 compute loses precision (full-GPU decode ~12 dB); this is fp16 precision, not a fusion bug (zoo README L1174).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
