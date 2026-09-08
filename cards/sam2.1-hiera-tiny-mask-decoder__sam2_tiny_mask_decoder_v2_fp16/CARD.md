---
family: sam2
license: apache-2.0
model_id: sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16
source_url: https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder
task: mask-generation
---

# sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16

| | |
|---|---|
| **Task** | mask-generation |
| **Family** | sam2 |
| **Source** | https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litert-samples interactive_segmentation conversion/convert_sam2_decoder.py) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python conversion/convert_sam2_decoder.py` |
| **Quantization** | fp16 (float_casting weight quantization; from the artifact name and script) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `sam2_tiny_mask_decoder_v2_fp16.tflite` | `80d668b19156c548f31a8c8eb3cc1da2127de36e24e96ef923467a1e54c99e76` | 16.182 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-13/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 398.045 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |
| webgpu_mldrift | output_mismatch | yes | MISMATCH | 0.026880677356045733 | 7.908 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0.dev20260804/2026-08-31/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16__raspberry-pi-5.json`, `data/device_runs/2.2.0/2026-08-26/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16__galaxy-s26.json`, `data/device_runs/2.2.0/2026-08-28/sam2.1-hiera-tiny-mask-decoder__sam2_tiny_mask_decoder_v2_fp16__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 10.548 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 8.045 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |
| pixel-8a | cpu_xnnpack | fallback | no | - | - | 222.813 | - | - | - | - | Pixel 8a · Tensor G3 · litert 2.2.0 · Android 16 | 2026-08-28 | measured |
| pixel-8a | gpu_mldrift | pass | yes | - | - | 30.875 | - | - | - | - | Pixel 8a · Tensor G3 · litert 2.2.0 · Android 16 | 2026-08-28 | measured |
| raspberry-pi-5 | cpu_xnnpack | pass | - | - | - | 158.27 | - | - | - | 135.48 | Raspberry Pi 5 Model B Rev 1.1 · litert 2.2.0.dev20260804 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-08-31 | measured |

## Pitfalls

- Why v2 exists: the v1 build's attention was written with the batch dim collapsed (q/k/v [heads,N,d], rank 3) — it compiles, fully delegates (358/358 LITERT_CL) and matches PyTorch on desktop, yet returns silently wrong masks on the Pixel 8a GPU (corr 0.265 vs CPU; not fp16 — fp32 GPU compute still 0.473). v2 keeps the leading batch dim ([1,heads,N,d], rank-4 SDPA): corr 0.9998 / binary-IoU 0.999 vs CPU and ~20% faster (6.8 vs 8.5 ms/tap) (HF card).
- The batchless miscompute is not Android-specific: on Mac Metal 2.1.6 v1 is rel 8.1e-02 vs v2 4.2e-07, and the browser sweep shows the same (v1 max_abs 4.81) — isolation probes cleared every op family individually, so it is a fusion/partition-level interaction on the batchless layout (rewrite-reach-survey 2026-08-13 addendum).
- Drop-in replacement: v2 inputs/outputs are identical to v1 (image_embeddings [1,256,64,64], sparse_prompt [1,2,256], feat_s1 [1,64,128,128], feat_s0 [1,32,256,256] -> masks [1,3,256,256] + iou [1,3]) at +45 RESHAPE +2 TRANSPOSE (HF card; survey addendum).
- Re-authoring in the script: ConvTranspose2d -> zero-stuff Conv2d; SafeLayerNorm (scale-before-square, fp16-overflow-safe); image positional embeddings + no-mask dense prompt baked constant; multimask path as a static slice (no argmax/gather) (convert_sam2_decoder.py docstring).
- The point prompt encoder runs host-side (sin/cos in Kotlin); pair with litert-community/SAM2.1-Hiera-Tiny-Image-Encoder run once per image (HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
