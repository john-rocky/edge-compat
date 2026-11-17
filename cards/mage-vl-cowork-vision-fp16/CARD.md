---
family: mage-vl
license: apache-2.0
model_id: mage-vl-cowork-vision-fp16
source_url: https://huggingface.co/litert-community/Mage-VL-Cowork
task: image-text-to-text
---

# mage-vl-cowork-vision-fp16

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | mage-vl |
| **Source** | https://huggingface.co/litert-community/Mage-VL-Cowork |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3 (the checkpoint's Qwen3-4B text model as a standalone Qwen3ForCausalLM with the GPU's cache-update and attention composites; the vision encoder and 2 x 2 merger for one static picture, its LayerNorm inputs scaled and its fully connected weights stored as fp16) and litert-lm-builder 0.16.1 as a fast_vlm bundle; repro = hf-to-litertlm scripts/ship_mage_vl_cowork.sh (card L309-L311, L325-L328) litert-torch 0.9.3, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, litert-lm-builder 0.16.1, transformers 5.14.1, torch 2.13.0, torchao 0.18.0 on Python 3.14.6 for the export; torch 2.12.1, transformers 5.14.1, tokenizers 0.22.2, safetensors 0.8.0 on Python 3.12.13 for the preparation; litert-lm 0.19.0 for the runs (recipe entry) |
| **Command** | `microsoft/Mage-VL-Cowork rev 6fbe7f91 -> PY_PREP=... PY_EXPORT=... PY_SAMPLE=... LITERT_LM=... bash scripts/reproduce_vlm.sh mage-vl-cowork of hf-to-litertlm (the key that also builds the sibling file; run as written on 2026-11-04, both files section by section the same as the shipped bytes)` |
| **Quantization** | the sibling's decoder, embedding table and adapter (text decoder int4, blockwise, 128 per block, OCTAV, a 4,096-token KV cache, fp16 on the GPU through an exact rotary-position path; int8 embedding table); a vision encoder with fp16 fully connected weights and biases and the model metadata reduced_precision_support = fp16accfp16 for XNNPACK's fp16 path on the CPU (card L31, L309-L311, L325) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Mage-VL-Cowork_vision-fp16.litertlm` | `1e977695d8f46222653233143e95431ccf6e46bfabe2c77073abf2e8ba75c5df` | 3051.369 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.0/2026-11-04/mage-vl-cowork-vision-fp16__galaxy-s26.json`, `data/device_runs/0.19.0/2026-11-04/mage-vl-cowork-vision-fp16__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 2988 | - | 34.1 | 5.16 | 87802.8 | 5518.4 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-04 | measured |
| galaxy-s26 | gpu | fallback | no | pass | sig:the GUI agent line (decoder on the GPU, vision encoder on the CPU): 120 ScreenSpot-v2 screenshots, one fresh process each, each click judged against the target's box; bar per group = the original model's lower rate (fp32 or bf16) minus 5 points | - | - | - | - | 2715.3 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-04 | measured |
| galaxy-s26 | gpu | fallback | no | - | sig:the long requests (decoder on the GPU, vision encoder on the CPU): the hybrid prompt as written (3,186 tokens) and three with 40 earlier-step lines (3,546-3,547 tokens) | - | - | - | - | 2561.8 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-04 | measured |
| galaxy-s26 | gpu | fallback | no | - | 1024 | - | 275.87 | 20.09 | 3771.5 | 1078.2 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-04 | measured |
| galaxy-s26 | gpu | fallback | no | - | 2988 | - | 248.5 | 17.6 | 12123.2 | 2547.8 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:12 of the line's screenshots with the decoder on the GPU and the vision encoder on the CPU (vision_backend=CPU, the section's FP16 metadata), one fresh process each; bar = 6 clicks in the box | - | - | - | - | 3161.3 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the hybrid prompt's two steps on 11 held-out requests (6 GUI, 5 shell) with the decoder on the GPU and the vision encoder on the CPU; the reply of the last inference judged | - | - | - | - | 3233.1 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-04 | measured |

## Pitfalls

- CPU path only: only the vision encoder on the CPU uses this file; its encoder was not run on a GPU. The decoder runs on the GPU and vision_backend = CPU (card L24, L313).
- Why fp16 on the CPU: the model metadata makes XNNPACK compute the encoder in fp16, which took the vision stage on the Galaxy S26 from 39.86 s to 18.24 s (warm, from a cool phone, these two encoders in builds with a shorter KV cache; on this file 18.83 and 19.08 s) (card L309).
- LayerNorm range: the LayerNorms of layers 13-23 take their input times 2^-k and their eps times 4^-k (k up to 4), because their squared differences pass fp16's largest value (65,504); the first LayerNorm takes its input times 8 and its eps times 64, because on a flat screen its variance is an fp16 subnormal, the fp16 ADD with eps returned 0, RSQRT gave inf and the attention spread the NaN to every token. Neither moves the encoder's fp32 output (bit-equal on 11 pictures) (card L311).
- Weights stored as fp16: the fully connected weights and biases are fp16 (a section of 613 MB against 1.22 GB), because LiteRT-LM 0.19.0's weight cache aborts the engine on the fp16 weights XNNPACK makes from fp32 ones unless the cache is off (card L311).
- Not the fp32 encoder's output: in fp16 on the CPU the vision output is close to fp32's, not equal (correlation 0.977-0.985 on three real screenshots, 0.53-0.55 on single-colour pictures), so this file has its own test rows (card L313).
- One picture size: the bundle takes one screenshot per request, stretched to 1920 x 1088 (68 x 120 patches, 2,040 image tokens), portrait phone screens included. At other sizes the original model misplaces its clicks: on the 40 desktop screenshots, 1344 x 768 was right on 3 against 21 at 1920 x 1088 (card L26, L249).
- fp16 positions on the GPU: fp16 holds integers exactly only up to 2,048, and a request's positions go past it (to 3,451 on the 120 test screenshots, to 3,900 with 40 earlier-step lines). The graph computes its rotary angles from position = 2048·h + l with precomputed tables for the 2048·h part (card L325).
- Galaxy S26: this file is the phone's form because the fp32 encoder on its GPU ends the process in the Adreno driver and the encoder on its NPU crashes the compute DSP; neither file has an NPU form (card L24, L317, L318).
- One after another the phone takes longer: 50.75 s from a cool phone against a median of 83.8 s; 108 of the 120 started at thermal status 3 and 118 under a CPU cap. MemAvailable fell by 4.95-5.10 GB in the three cool-phone processes, to 2.21 GB at the lowest (card Performance table, L298, L299).
- Everything on the Galaxy S26's CPU is slower and it ran hot: 159.7-182.7 s, right after a GPU run with no wait, at thermal status 3 and under different CPU caps, so those two do not compare the caches (card L33, L300).
- Pixel 11 Pro: this file ran there only in the Android app on the hybrid prompt, a vision stage of 35.6 s from thermal status 0 and 90.8 s on the next inference; it did not take the 120-screenshot test on that phone (card L301).
- The hybrid prompt's first reply is usually `screenshot` although the screenshot is in the request, so a step takes one or two inferences. A new conversation holding the reply and the screenshot is the second request; prompt and reply together reached 3,737 of the 4,096 tokens. On 11 requests on drawn screens the right tool came on 10 and a click inside the target on 4 of 6, on the Galaxy S26 in an Android app and on the Mac (card L132-L145).
- Later turns: from the second turn on LiteRT-LM puts the previous answer's stop token into the KV cache with a zero embedding (this bundle's decoder takes embeddings), read in the source of 0.19.0 and measured with 0.18.1 on other models, not on these files; a new conversation per inference avoids it (card L332).
- Sampling: the file carries no sampler settings, so LiteRT-LM 0.19.0 samples with top-k 1 on the CPU and the GPU; every result is greedy, while the model card's own setting is temperature 0.7 and top_p 0.8 (card L216).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
