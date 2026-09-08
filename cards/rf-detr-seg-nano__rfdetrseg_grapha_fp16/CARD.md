---
family: rf-detr
license: apache-2.0
model_id: rf-detr-seg-nano__rfdetrseg_grapha_fp16
source_url: https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT
task: image-segmentation
---

# rf-detr-seg-nano__rfdetrseg_grapha_fp16

| | |
|---|---|
| **Task** | image-segmentation |
| **Family** | rf-detr |
| **Source** | https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 |
| **Command** | `python scripts/build_rfdetrseg_split.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `54b36190cac9da79da0fe4ce3e80b8acecd8b12429c227ac4fa563cbbb7d6d89.tflite` | `1b85a919c19a1832b729f546c3f60ceca179b669258d7ab244fac96890bf630a` | 44.85 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-20/rf-detr-seg-nano__rfdetrseg_grapha_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2194.563 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 0.19101343154907227 | 24.917 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-20 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/rf-detr-seg-nano__rfdetrseg_grapha_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/rf-detr-seg-nano__rfdetrseg_grapha_fp16__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 35.283 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 27.833 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 695.348 | - | - | - | 182.08 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- The cls+pos embedding and patch pos-embed are HOST-FED inputs (clspos.bin, pospatch.bin published beside the model): the GPU delegate mis-executes compute chains that consume large baked-constant tensors (fp32 identical wrong numbers) — HF card 'baked-constant execution bug' section.
- memory is emitted as memory*2: a [1,N,C] tensor that is both consumed in-graph and a graph output comes back zeroed on the delegate; the host halves it — HF card.
- The two-stage query selection (top-100 + gather + reparam with refpoint_embed.bin) runs on the host between graph A and graph B — HF card pipeline diagram.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
