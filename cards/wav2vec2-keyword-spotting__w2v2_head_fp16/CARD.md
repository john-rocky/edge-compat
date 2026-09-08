---
family: wav2vec2
license: apache-2.0
model_id: wav2vec2-keyword-spotting__w2v2_head_fp16
source_url: https://huggingface.co/litert-community/wav2vec2-keyword-spotting
task: audio-classification
---

# wav2vec2-keyword-spotting__w2v2_head_fp16

| | |
|---|---|
| **Task** | audio-classification |
| **Family** | wav2vec2 |
| **Source** | https://huggingface.co/litert-community/wav2vec2-keyword-spotting |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python wav2vec2-kws/scripts/build_w2v2_split.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `7847d1511bcf082db8cb479406177fea8433895539c1a10f97dc22f6ffbddad9.tflite` | `8a397899a1cc9d948aca3512a56b14a14347cb9cf9d51f486ff926d8819f010c` | 172.201 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/wav2vec2-keyword-spotting__w2v2_head_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 685.443 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.0005259458445152539 | 12.15 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/wav2vec2-keyword-spotting__w2v2_head_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/wav2vec2-keyword-spotting__w2v2_head_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 9.197 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 5.668 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 106.275 | - | - | - | 600.39 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The model ships as two graphs (frontend + head) because the full 1008-node graph exceeds the Mali shader-compile limit; both halves are needed for a prediction (zoo README L1195).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
