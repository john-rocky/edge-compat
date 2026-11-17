---
family: mage-vl
license: apache-2.0
model_id: mage-vl-cowork
source_url: https://huggingface.co/litert-community/Mage-VL-Cowork
task: image-text-to-text
---

# mage-vl-cowork

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | mage-vl |
| **Source** | https://huggingface.co/litert-community/Mage-VL-Cowork |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (the checkpoint's Qwen3-4B text model as a standalone Qwen3ForCausalLM with the GPU's cache-update and attention composites; the vision encoder and 2 x 2 merger for one static picture) and litert-lm-builder 0.16.1 as a fast_vlm bundle; repro = hf-to-litertlm scripts/ship_mage_vl_cowork.sh (card L325-L328) litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.16.1, transformers 5.14.1, torch 2.13.0, torchao 0.18.0 on Python 3.14.6 for the export; torch 2.12.1, transformers 5.14.1, tokenizers 0.22.2, safetensors 0.8.0 on Python 3.12.13 for the preparation; litert-lm 0.19.0 for the runs (recipe entry) |
| **Command** | `microsoft/Mage-VL-Cowork rev 6fbe7f91 -> PY_PREP=... PY_EXPORT=... PY_SAMPLE=... LITERT_LM=... bash scripts/reproduce_vlm.sh mage-vl-cowork of hf-to-litertlm (the key that also builds the sibling file; run as written on 2026-11-04, both files section by section the same as the shipped bytes)` |
| **Quantization** | text decoder int4 (blockwise, 128 per block, OCTAV), one prefill signature of 1,024 tokens and the decode signature with a 4,096-token KV cache, no activation type declared = fp16 on the GPU through an exact rotary-position path; int8 embedding table outside the decoder; vision encoder and adapter fp32 weights, fp32 activations declared (card L30, L325, L326) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Mage-VL-Cowork.litertlm` | `0d922aa53f65b40e5073e034f7c06c098389002ab4ad9c4f99a5f3654dd77b14` | 3627.728 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.0/2026-11-04/mage-vl-cowork__mac-studio-m4-max.json`, `data/device_runs/0.19.0/2026-11-05/mage-vl-cowork__iphone-18-pro.json`, `data/device_runs/0.19.0/2026-11-05/mage-vl-cowork__pixel-11-pro.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iphone-18-pro | cpu | pass | - | - | 2988 | - | 81.5 | 10.5 | 60814.0 | 3616.0 | iPhone 18 Pro (iPhone19,2) · litert-lm 0.19.0 · iOS 27.2 (24B5099f) | 2026-11-04 | measured |
| iphone-18-pro | gpu | pass | - | pass | sig:the GUI agent line (decoder and vision encoder on Metal): 120 ScreenSpot-v2 screenshots, one fresh process each, each click judged against the target's box; bar per group = the original model's lower rate (fp32 or bf16) minus 5 points | - | - | - | - | 4379.0 | iPhone 18 Pro (iPhone19,2) · litert-lm 0.19.0 · iOS 27.2 (24B5099f) | 2026-11-05 | measured |
| iphone-18-pro | gpu | pass | - | - | sig:the long requests (decoder and vision encoder on Metal): the hybrid prompt as written (3,186 tokens) and three with 40 earlier-step lines (3,546-3,547 tokens) | - | - | - | - | 4140.0 | iPhone 18 Pro (iPhone19,2) · litert-lm 0.19.0 · iOS 27.2 (24B5099f) | 2026-11-04 | measured |
| iphone-18-pro | gpu | pass | - | - | 2988 | - | 514.2 | 35.26 | 19361.0 | 4321.0 | iPhone 18 Pro (iPhone19,2) · litert-lm 0.19.0 · iOS 27.2 (24B5099f) | 2026-11-04 | measured |
| mac-studio-m4-max | cpu | pass | - | pass | sig:12 of the line's screenshots on the CPU (decoder and vision encoder, XNNPACK, 4 threads), one fresh process each; bar = 6 clicks in the box | - | - | - | - | 3902.4 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 117.8 | 17.16 | 8751.5 | 6687.7 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 2988 | - | 107.3 | 19.71 | 27913.3 | 3903.8 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the GUI agent line (decoder and vision encoder on the GPU): 120 ScreenSpot-v2 screenshots, one fresh process each, each click judged against the target's box; bar per group = the original model's lower rate (fp32 or bf16) minus 5 points | - | - | - | - | 5193.7 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:the long requests (decoder and vision encoder on the GPU): the hybrid prompt as written (3,186 tokens) and three with 40 earlier-step lines (3,546-3,547 tokens) | - | - | - | - | 5258.0 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1352.41 | 94.52 | 767.8 | 3945.4 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 2988 | - | 1222.0 | 89.24 | 2457.1 | 5198.5 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| pixel-11-pro | gpu | fallback | no | - | sig:one request with the vision encoder on the CPU (decoder on the GPU): desktop_006, 2,988 tokens in, a fresh process each | - | 130.7 | 7.2 | 23008.6 | 2447.3 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-04 | measured |
| pixel-11-pro | gpu | fallback | no | pass | sig:the GUI agent line (decoder and vision encoder on the GPU): 120 ScreenSpot-v2 screenshots, one fresh process each, each click judged against the target's box; bar per group = the original model's lower rate (fp32 or bf16) minus 5 points | - | - | - | - | 1256.8 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-05 | measured |
| pixel-11-pro | gpu | fallback | no | - | sig:the long requests (decoder and vision encoder on the GPU): the hybrid prompt as written (3,186 tokens) and three with 40 earlier-step lines (3,546-3,547 tokens) | - | - | - | - | 1257.2 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-04 | measured |
| pixel-11-pro | gpu | fallback | no | - | 2988 | - | 120.2 | 7.81 | 25097.2 | 1256.2 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-04 | measured |

## Pitfalls

- One picture size: the bundle takes one screenshot per request, stretched to 1920 x 1088 (68 x 120 patches, 2,040 image tokens), portrait phone screens included. At other sizes the original model misplaces its clicks: a drawn button at 9 places was hit 9 of 9 times at 1920 x 1088 and at most 4 of 9 at six other sizes; on the 40 desktop screenshots, 1344 x 768 was right on 3 against 21 at 1920 x 1088 (card L26, L249).
- fp16 positions on the GPU: fp16 holds integers exactly only up to 2,048, and a request's positions go past it (to 3,451 on the 120 test screenshots, to 3,900 with 40 earlier-step lines). The graph computes its rotary angles from position = 2048·h + l with precomputed tables for the 2048·h part. On 12 test screenshots the fp16 decoder put the click in the target's box on 1 without that change (0 with the GPU composites added) and on 8 with it; the same decoder declared fp32 (before the composites) did on 7 (card L325).
- Vision activations on the GPU: this file declares fp32. With fp16 activations on the Mac GPU the encoder's output correlates at 0.599 with the CPU's fp32 output (card L326).
- Galaxy S26: not this file. Its vision encoder, run alone with a LiteRT 2.2.0 CompiledModel, compiles with all 7,724 nodes on the GPU, and about 8 s later the Adreno driver ends the process (`cl_a8x_cmdbuf_mgr_submit_ibs`, exit code 137) with 3.88 GB of memory still available; attention chunks of 1,020 rows instead of 255 end the same way. The whole file was not run in this form; the phone takes the sibling file (card L24, L317).
- NPU: the encoder compiled for SM8850 (QAIRT 2.47, O2) loads, and its first execution crashes the compute DSP (`SSR`) on both graphs tried. Neither file has an NPU form (card L318).
- Pixel 8a: neither file was run on it; two earlier builds with a shorter KV cache stopped while the engine was being created, at the test's guard on MemAvailable under 400,000 kB (card L319).
- Replies differ by device and, on the Pixel 11 Pro, by process: the iPhone and the Mac GPU run this file and their replies were equal on none of the 120 screenshots (the same click point on 28). On the Pixel 11 Pro desktop_006 gave one reply on 3 cold processes and 4 different replies on 4 warm ones; cause not read (card L248).
- One after another the phones take longer: the iPhone 26.16 s from a cool phone against a median of 58.73 s, the Pixel 11 Pro 106.97 s after a 7-minute rest against 148.8 s; 107 of the iPhone's 120 started and ended at thermal state serious, 92 of the Pixel 11 Pro's started at thermal status 3 (card Performance table, L298).
- The hybrid prompt's first reply is usually `screenshot` although the screenshot is in the request, so a step takes one or two inferences; the original model does the same on the two tasks it was run on. A new conversation holding the reply and the screenshot is the second request; prompt and reply together reached 3,737 of the 4,096 tokens (card L132-L136).
- Later turns: from the second turn on LiteRT-LM puts the previous answer's stop token into the KV cache with a zero embedding (this bundle's decoder takes embeddings), read in the source of 0.19.0 and measured with 0.18.1 on other models, not on these files; a new conversation per inference avoids it (card L332).
- Sampling: the file carries no sampler settings, so LiteRT-LM 0.19.0 samples with top-k 1 on the CPU and the GPU; every result is greedy, while the model card's own setting is temperature 0.7 and top_p 0.8 (card L216).
- Reading the reply: the runtime returns the whole reply as text and no tools are passed to it; read the first <tool_call> yourself and leniently: with four earlier steps the original model wrote a typographic opening quote before `arguments` in 2 of 40 replies, none of the 120 test replies did (card L214).
- Memory: the iPhone's app has the increased-memory-limit entitlement (limit read 6,144 MiB; at least 2,004 MiB stayed free) and these files were not run without it. On the Pixel 11 Pro, a phone with 16 GB, MemAvailable fell by 7.23-7.47 GB in the warm and cold rows (card L299).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
