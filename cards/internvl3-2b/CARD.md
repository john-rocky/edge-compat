---
family: internvl3
license: apache-2.0
model_id: internvl3-2b
source_url: https://huggingface.co/litert-community/InternVL3-2B
task: image-text-to-text
---

# internvl3-2b

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | internvl3 |
| **Source** | https://huggingface.co/litert-community/InternVL3-2B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (LiteRT-LM fast_vlm bundle: VISION_ENCODER + VISION_ADAPTER + single-token EMBEDDER + PREFILL_DECODE with embeddings input) (HF card Conversion notes) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (the HF card states the bundle layout and rewrites, not the invocation)` |
| **Quantization** | TODO — the HF card states no quantization recipe for this bundle; sections: VISION_ENCODER [1,448,448,3]->[1,256,4096], VISION_ADAPTER [1,256,4096]->[1,256,1536], externalized embedder, PREFILL_DECODE (HF card Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `InternVL3-2B.litertlm` | `cb7d63cbf2f5d9b3eb307012b54dc4fc4d68fef78dc3a5d4fbba41c59812f9b6` | 1363.411 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/internvl3-2b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/internvl3-2b__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 239 | - | 207.94 | 43.67 | 1170.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 203 | - | 573.22 | 34.4 | 380.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |

## Pitfalls

- Known limitation — one image per conversation on the GPU backend: on the GPU (Metal) backend a second image in the same conversation truncates the answer; on the CPU backend multi-image works (verified); the truncation reproduces with other fast_vlm models including litert-community/FastVLM-0.5B (ruled out as causes: max_num_images, and the vision encoder's 5D reshape — a 4D-clean rebuild still truncates on GPU); single-image VQA on the iPhone 17 Pro GPU runs ~45 tok/s, so it is the runtime's GPU fast_vlm path, not this bundle. For reliable multi-image, run on the CPU backend (HF card Known limitation).
- The vision encoder bakes ImageNet normalization + the NCHW transpose into the graph (the runtime feeds a [0,1] NHWC image), and the InternViT attention is rewritten 4D-clean (qkv split before the head reshape — no GPU-rejected 5D reshape), numerically identical (corr ≈ 1.0) (HF card Conversion notes).
- Decoder exported with an externalized embedder; InternVL's dynamic-NTK rope_scaling is stripped to base RoPE, valid since the export cache <= the base context window (HF card Conversion notes).
- License: MIT (the InternVL model) + Apache-2.0 (the Qwen2.5 language component); converted artifacts released under the same terms (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
