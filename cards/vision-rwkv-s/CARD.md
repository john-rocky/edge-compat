---
family: vision-rwkv
license: apache-2.0
model_id: vision-rwkv-s
source_url: https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT
task: image-classification
---

# vision-rwkv-s

| | |
|---|---|
| **Task** | image-classification |
| **Family** | vision-rwkv |
| **Source** | https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python vrwkv/scripts/build_vrwkv.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `3b30dd130a0d0e53caa8ddb188f9749a64fb45c53486bc398b79594d5bbfa52f.tflite` | `8015269594520f50e4ce6fb3bf4812a79ec5414fc66373370f5e2739ef545b4d` | 45.987 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/vision-rwkv-s.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 316.283 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 24.659357170412783 | 27.517 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/vision-rwkv-s__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 1082.624 | - | - | - | 277.05 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The second input is the constant token-distance matrix dist[1,1,196,196] (dist[t,i]=|t-i|), deliberately fed at runtime so the [C,T,T] decay bias is not const-folded into an unshippable 1.5 GB flatbuffer (zoo README L1481-1483).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
