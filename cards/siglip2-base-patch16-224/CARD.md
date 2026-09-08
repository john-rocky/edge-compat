---
family: siglip
license: apache-2.0
model_id: siglip2-base-patch16-224
source_url: https://huggingface.co/litert-community/SigLIP2-base-patch16-224
task: image-feature-extraction
---

# siglip2-base-patch16-224

| | |
|---|---|
| **Task** | image-feature-extraction |
| **Family** | siglip |
| **Source** | https://huggingface.co/litert-community/SigLIP2-base-patch16-224 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litertlm-convert scripts/convert_siglip2.py; timm image tower) 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python scripts/convert_siglip2.py` |
| **Quantization** | fp16 (float_casting weight quantization; single-graph 185 MB file) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `siglip2_base_224_fp16.tflite` | `a30ebb7b3ee15eaa68a18f9ab6a2ed740c15c343d25d898dc482317473320854` | 176.847 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-13/siglip2-base-patch16-224.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 2124.707 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |
| webgpu_mldrift | pass | yes | pass | 0.0008081583214905942 | 19.635 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-13 | measured |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-08-26/siglip2-base-patch16-224__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu_mldrift | pass | - | - | - | 13.704 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · Android 16 | 2026-08-26 | measured |
| galaxy-s26 | npu_qnn | pass | - | - | - | 6.943 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.2.0 · QAIRT (Hexagon, JIT on-device) · Android 16 | 2026-08-26 | measured |

## Pitfalls

- Input is NCHW [1,3,224,224] normalized to [-1,1] ((x/255-0.5)/0.5) BY THE CALLER; output [1,768] is already L2-normalized (HF card).
- Re-authoring (script docstring): fused qkv 5-D head-split decomposed to separate q/k/v with 4-D SDPA; the attention-pool's const-latent batch-matmul expressed as broadcast-multiply + reduce-sum (const@non-const BMM is rejected/mis-computed); overflow-safe LayerNorm because the delegate reduces variance in fp16 even for an fp32 graph (deep-ViT activations overflow 65504).
- Full GPU residency on Pixel 8a: 809/809 nodes LITERT_CL, no Flex ops, GPU output corr ~1.0 vs PyTorch (HF card).
- Zero-shot classification needs the TEXT tower host-side: open_clip ViT-B-16-SigLIP2 embeddings (prompt 'This is a photo of {label}.'), dot product on device (HF card).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
