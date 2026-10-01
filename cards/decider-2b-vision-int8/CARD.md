---
family: decider
license: apache-2.0
model_id: decider-2b-vision-int8
source_url: https://huggingface.co/litert-community/decider-2b-vision-LiteRT
task: image-text-to-text
---

# decider-2b-vision-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | decider |
| **Source** | https://huggingface.co/litert-community/decider-2b-vision-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch decoder export (fork john-rocky/litert-torch @115a13607c730c81018bb9789138a3e5e5119e3d + the qwen35 hybrid patch 0a01e2ae…, with the derived-M-RoPE rotary of scripts/mrope_derived.py installed only inside the export process) + litert-torch / litert-converter vision encoder and merger export at 256x256 + ai-edge-quantizer weight forms + litert-lm-builder bundle (build_bundle_g256.py) + add_executor_metadata.py (repro = hf-to-litertlm decider2bv_work/, REPRODUCE.md rounds 2-4) decoder: litert-torch 0.9.2 (patched fork clone), torch 2.12.1, transformers 5.14.1, litert-converter 0.3.0, ai-edge-litert 2.1.6, ai-edge-quantizer 0.8.0 (venv-export lock); vision: litert-torch 0.9.3, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1 with transformers 5.17.0's position-embedding taps; bundle and runtime checks: litert-lm-builder / litert-lm 0.17.1, ai-edge-litert 2.2.0 (venv-readout lock); reference = the checkpoint's own decider/vision.py in fp32 on CPU, torch 2.14.0, transformers 5.17.0 (REPRODUCE.md section 1) |
| **Command** | `export_decoder_g256.py --out out/decoder_g256_r4_fp32 --step-form relu_diff (prefill ladder 1024/256/64/16/4/1 + decode, cache 4096, externalized single-token embedder, 48 state buffers) -> quantize_r4.py dyn8 decoder / embedder (recipe wi8fc) -> bundle_r4.sh dyn8 (build_bundle_g256.py with the round-3 fp16 vision encoder + adapter from quantize_fp16_r3.py: fast_vlm, image 256x256, max_num_tokens 4096, identity jinja template, no start token, stop token 248044, the snapshot's tokenizer.json, prefer_activation_type fp32 on the decoder section; then add_executor_metadata.py for the 48 state buffers, 36 linear-attention + 12 K/V) (REPRODUCE.md rounds 3-4)` |
| **Quantization** | dynamic int8 CHANNELWISE on every decoder FULLY_CONNECTED (activations quantized on the fly on the CPU), token embedding int8 CHANNELWISE (recipe wi8fc); vision encoder and adapter fp16 float casting; fp32 activation preference declared on the decoder section (Hub card Files; REPRODUCE.md round 4) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `decider-2b-vision_int8.litertlm` | `5fb2e19aa2066d2e3955431366bc572edb7abbc6037db53d4eaf4a4a4d0e7bd9` | 3024.179 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-29/decider-2b-vision-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-29/decider-2b-vision-int8__galaxy-s26.json`, `data/device_runs/0.17.1/2026-09-29/decider-2b-vision-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | sig:Android app (hfmodels-android samples/ask, release build); one conversation, image + 5 questions | - | - | - | - | 3832.38 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-29 | measured |
| galaxy-s26 | cpu | fallback | no | - | 150 | - | 136.68 | - | 1220.0 | 4540.08 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-29 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/ask, release build); one conversation, image + 5 questions | - | - | - | - | 3874.18 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 | 2026-09-29 | measured |
| galaxy-s26 | gpu | pass | yes | - | 150 | - | 340.76 | - | 560.0 | 4321.35 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-29 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-09-29 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-09-29 | measured |

## Pitfalls

- Use the GPU when the probabilities matter: on the CPU this file quantizes activations on the fly and its probabilities move against upstream fp32 on the published image questions (51/53 answers equal, max |dp| 0.429, p95 0.164), and they also depend on how the prompt is split into prefill chunks (max |dp| 0.233 with one padded chunk); on the Mac GPU (Metal CompiledModel, one padded chunk per answer slot) it computes the int8 weights in float: 52/53, max |dp| 0.032, p95 0.011. Probabilities on the phone were not measured, because the runtime returns text only (Hub card Files note; FINDINGS section 5).
- Galaxy S26 (SM-S942Q, MemTotal 11,389,756 kB, LiteRT-LM v0.16.0 litert_lm_advanced_main, a different runtime version from the Mac rows): the first token was upstream's answer letter on 6/6 published rows on the CPU and on the GPU, with prefill counts equal to upstream's and no 'Validation error' lines. Speed row (game_pong_atari_level, 150 prompt tokens, greedy, cold, n = 1): CPU 4 threads TTFT 1.22 s, prefill 136.68 tok/s, engine creation 17.7 s (+2.0 s conversation set-up), peak VmHWM 4.54 GB; GPU (OpenCL) 0.56 s, 340.76 tok/s, 64.4 s (+7.0 s), 4.32 GB (Hub card Performance).
- On the S26 GPU all 7 decoder signatures were fully delegated to OpenCL (one partition each); the vision encoder ran on the GPU and the adapter on the CPU. VmHWM does not include GPU memory: during the int8 GPU run the phone's MemAvailable fell to 0.77 GB, so the int8 GPU run is close to the limit of a phone of this RAM class; the other five int8 CPU rows peaked at 4.82-4.85 GB VmHWM (Hub card Performance).
- Mac (LiteRT-LM 0.17.1): the int8 file was not timed; its first token was upstream's letter on 12/12 single-question image rows on the CPU and on the GPU (11/12 equal to this file's own CPU graph), and synth_center_circle gave the reference answer B on both runtime legs while the int8 CPU graph gives A under both prefill feedings; the runtime's own chunk plan is not observed (Hub card Performance; FINDINGS sections 5 and 8).
- The model returns option probabilities, not text: upstream reads the letter logits A-J at one answer slot per question (a ' (' token after ':' with 'Answer' within the 5 tokens before it), cuts them to the option count and softmaxes at T = 1. Through LiteRT-LM only the answer letter is available: the greedy first token of one question per request, one image first, and it is the top token over the whole vocabulary, not only over the option letters (on one two-question published row the top token for a two-option question is 'C'). LiteRT-LM's text-scoring call cannot stand in for the probabilities (its scores disagree with its own greedy decode on 0.17.1, LiteRT-LM #3561). Probabilities, text-only requests and several questions per request go through the bundled reference readout, reference/decider_litert.py (ai-edge-litert 2.2.0, numpy, Pillow, tokenizers; no torch, transformers or litert-lm) (Hub card Use it, Limitations).
- Fixed 256x256 input (64 image tokens) where upstream picks a resolution per image: for 224x224, 256x240 and 256x256 originals upstream's own processing gives the same pixels (13 published requests, 21 slots, |dp| 0); every other size gets fewer image tokens than upstream would use (70 for the 160x210 frames, 144 to 475 for the larger synthetic images), and upstream fp32 itself then matches its original-image answer on 30/32 published slots, max |dp| 0.373, p95 0.318, up to 0.58 on photographs and documents kept out of the repo. Resize to 256x256 with PIL bicubic before sending; the runtime's own image resize was never exercised (Hub card The price of the fixed 256x256 input, Limitations; FINDINGS section 2).
- Positions: the checkpoint uses 3-channel M-RoPE (image tokens on an 8x8 grid) and LiteRT-LM passes one 1-D position, so the decoder derives the three channels inside its graph with element-wise ops; plain 1-D positions flipped 2 of the 53 published image answers and moved one probability by 0.198 in upstream fp32. Text-only requests must start at position 65, where the derived positions keep upstream's relative positions; the runtime numbers them from 0 (the 9 text-only questions moved by up to 0.0497 with one flip in the fp32 graph), so text-only requests go through the reference readout (Hub card What matches upstream 4-5; FINDINGS sections 1 and 7).
- Rebuilders: do not export the derived rotary's step as clamp(x, 0, 1). The converter lowers it to RELU_0_TO_1 (9 per signature), which the WebGPU delegate of LiteRT-LM 0.17.1 on the Mac refuses ('RELU_0_TO_1: Not supported op', 25,675 of 25,927 prefill_1024 ops on the GPU, 'Hint fully delegated to single delegate is set, but the graph is not fully delegated', engine creation fails, twice). The shipped decoders write it as relu(x) - relu(x - 1) (equal on integer inputs; fp32 letter logits bit-equal) and create the full-GPU engine on the Mac and, for this file, on the S26 OpenCL delegate (Hub card Performance; FINDINGS section 4).
- XNNPACK cache and GPU caches: LiteRT-LM's CPU cache folder holds 1,896,902,288 B for this decoder plus 1,312,687,576 B for the vision encoder and adapter; with the GPU backend the decoder program cache in a cache folder grew by 270,663,680 B at every engine creation in the lane's Mac runs, and a warm folder did not shorten GPU engine creation (Hub card Files; FINDINGS section 6 and Corrections).
- Bundle metadata shared by all three files: identity template (no role markers), no start token, stop token 248044, one 256x256 image, a 4096-token cache (rows longer than 4096 tokens do not fit; prompts above 2048 tokens were not exercised, and the derived rotary computes positions in float), ExecutorMetadata for the 48 state buffers and an fp32 activation preference for the decoder (no runtime log line states the precision the GPU executor used). One image per request, English only; the text behaviour is upstream's (decider-2b v5 text weights) (Hub card Files, Limitations; FINDINGS section 8 and Open questions).
- License Apache-2.0 as declared by the upstream checkpoint (source revision 863e290863655f1d6b69324d77d09ac972d21609); changes: text decoder, vision encoder and merger converted to LiteRT flatbuffers, M-RoPE derived in the decoder graph from a 1-D position, image fixed at 256x256, weights cast as listed, chat template replaced with an identity template, tokenizer repackaged unchanged, ExecutorMetadata and an fp32 activation preference added; the multi-token-prediction head is not included; community conversion, not an official Mapika or Qwen release (Hub card License and changes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
