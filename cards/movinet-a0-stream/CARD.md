---
family: movinet
license: apache-2.0
model_id: movinet-a0-stream
source_url: https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT
task: video-classification
---

# movinet-a0-stream

| | |
|---|---|
| **Task** | video-classification |
| **Family** | movinet |
| **Source** | https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python movinet/scripts/build_movinet.py` |
| **Quantization** | none (float32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `64841a4c8419d448019b54731d6d0dc77d6239d4d0c1a8e2a0415545377f56c7.tflite` | `0565d6e2173caf3e144e6a0a6461edfef9dc5cccacf9a093f311f8a8f05434c8` | 14.431 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/movinet-a0-stream.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 7.13 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | output_mismatch | no | MISMATCH | 0.04193414820059219 | 13.255 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/movinet-a0-stream__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 4.628 | - | - | - | 137.5 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- All recurrent-state plumbing is host-side: the stream-buffer shift register and pool running-sum accumulation are done by the caller; the graph consumes state and emits only fresh per-frame tensors (zoo README L400-402).
- Each emitted stream frame is decoupled from its compute use by a multiply against the runtime 1.0 input (input[46]) to dodge three stated silent Mali CompiledModel bugs (zoo README L402).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
