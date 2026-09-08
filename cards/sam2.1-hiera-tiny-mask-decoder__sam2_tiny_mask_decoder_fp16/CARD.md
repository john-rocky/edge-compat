---
family: sam2
license: apache-2.0
model_id: sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16
source_url: https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder
task: mask-generation
---

# sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16

| | |
|---|---|
| **Task** | mask-generation |
| **Family** | sam2 |
| **Source** | https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litert-samples interactive_segmentation conversion pipeline, pre-fix build) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python conversion/convert_sam2_decoder.py` |
| **Quantization** | fp16 (float_casting weight quantization) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `sam2_tiny_mask_decoder_fp16.tflite` | `8fc29cfa6e41741e03d8dbc53c18c945d5f98c8eb4a07376390022cc91d91ad5` | 16.172 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-13/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 396.69 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 10099.350182687609 | 8.33 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-08-28/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_fp16__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 10.077 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 8.08 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| pixel-8a | cpu_xnnpack | fallback | no | - | - | 218.668 | - | - | - | - | Pixel 8a · Tensor G3 · litert 2.2.0 · Android 16 | 2026-08-28 | measured |
| pixel-8a | gpu_mldrift | pass | yes | - | - | 30.055 | - | - | - | - | Pixel 8a · Tensor G3 · litert 2.2.0 · Android 16 | 2026-08-28 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 156.218 | - | - | - | 135.48 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- SUPERSEDED — use sam2_tiny_mask_decoder_v2_fp16.tflite. This build is the documented specimen of 'fully delegated yet silently wrong': 358/358 LITERT_CL nodes, banned ops NONE, desktop parity corr 1.0, but Pixel 8a GPU masks at corr 0.265 vs CPU (fp32 GPU compute still 0.473); on Mac Metal 2.1.6 rel 8.1e-02; browser sweep max_abs 4.81 (HF card; rewrite-reach-survey addendum).
- Root cause: attention exported with the batch dim collapsed (q/k/v [heads,N,d], rank 3); the delegate mis-computes the batchless layout at the fusion/partition level (isolation probes cleared every op family individually). Keep an explicit batch dim through attention blocks (survey addendum).
- Kept on the HF repo for reference/reproduction of the miscompute; CPU execution is correct (HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
