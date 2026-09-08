---
family: cohere-compass
license: apache-2.0
model_id: north-micro-vision-instruct-wi8
source_url: https://huggingface.co/litert-community/North-Micro-Vision-Instruct
task: image-text-to-text
---

# north-micro-vision-instruct-wi8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | cohere-compass |
| **Source** | https://huggingface.co/litert-community/North-Micro-Vision-Instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (hf-to-litertlm reproduce_vlm.sh key `north-micro-vision`; decoder export_hf + hand-assembled fast_vlm bundle via litert-lm-builder) litert-torch 0.9.3 (ai-edge-quantizer 0.8.0, litert-lm-builder 0.16.0; vision/prep on transformers 5.16.0.dev0) |
| **Command** | `bash scripts/ship_northmv.sh   # = prep_northmv_decoder.py (Cohere2 re-host, rope patch) -> export_northmv_decoder.py (RECIPE=dynamic_wi8_afp32, CACHE=4096, PREFILL=128,512,1024, externalize_embedder) -> convert_northmv_vision.py (IMG=512 DEEPSTACK=fold) -> build_northmv_bundle.py (TOK=hf, VENC/VADP=int8dyn, DEC_ACT=fp32_fp16)` |
| **Quantization** | decoder int8 dynamic-range weights (dynamic_wi8_afp32) with prefer_activation_type=fp32_fp16 declared on the PREFILL_DECODE section; embedder int8; vision encoder + adapter int8 dynamic-range |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `North-Micro-Vision-Instruct_wi8.litertlm` | `83e330f05c9077498324c7513f3786f953d9901734edb845f1155f72068b17fd` | 2929.052 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-19/north-micro-vision-instruct-wi8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-24/north-micro-vision-instruct-wi8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-08-19/north-micro-vision-instruct-wi8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 199 | - | 280.01 | 10.4 | 810.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 243.92 | 13.64 | 685.0 | 3046.316 | iPhone 17 Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-19 | measured |
| pixel-8a | gpu | pass | yes | - | 271 | - | 124.19 | 4.3 | 2410.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.1 | 2026-08-19 | measured |

## Pitfalls

- fast_vlm carries ONE image embedding; this model's three DeepStack vision embeddings (HF injects them after decoder layers 0/1/2) are folded into it in the adapter. Exactly representable (corr 1.0), but greedy-exact parity with the released model is not achievable — gate on teacher-forced top-1 (0.96 fold / 0.93 fold+1-D) and generation reading (FINDINGS.md).
- The decoder is re-hosted as Cohere2ForCausalLM with its rope patched to the checkpoint's Llama-style half-split layout; stock Cohere2 rope lowers to BROADCAST_TO + a 5-D CONCATENATION that the GPU delegate rejects (FINDINGS.md).
- Mali OpenCL fp16 decoder answers image turns as if blind (echoes the question) while text-only is perfect — fp16 accumulation overflows at the 256 image-token positions; the bundle declares prefer_activation_type=fp32_fp16 on the decoder section, which fixes it without runtime flags (measured Pixel 8a 2026-08-19).
- fp16 vision encoder compiled on Mali OpenCL hard-reboots the phone (twice, same stage); vision ships int8. Decoder on the CPU of an 8 GB phone pages against the 2.5 GB decoder (0.6 tok/s) — run the decoder on the GPU (FINDINGS.md).
- Tokenizer: the HF tokenizer.json is bundled; the sentencepiece conversion crashes on null-byte pieces and splits digits differently from Cohere's per-digit pre-tokenizer (FINDINGS.md).
- Runtime contract: 1-D positions replace M-RoPE — 2-D table cross-cell questions and digit-dense OCR degrade; describe/VQA/spatial preserved (HF card note).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
