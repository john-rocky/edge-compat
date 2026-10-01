---
family: decider
license: apache-2.0
model_id: decider-2b-vision-fp16
source_url: https://huggingface.co/litert-community/decider-2b-vision-LiteRT
task: image-text-to-text
---

# decider-2b-vision-fp16

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
| **Command** | `export_decoder_g256.py --out out/decoder_g256_r4_fp32 --step-form relu_diff (prefill ladder 1024/256/64/16/4/1 + decode, cache 4096, externalized single-token embedder, 48 state buffers) -> quantize_r4.py fp16 decoder / embedder (recipe wfp16) -> bundle_r4.sh fp16 (build_bundle_g256.py with the round-3 fp16 vision encoder + adapter from quantize_fp16_r3.py: fast_vlm, image 256x256, max_num_tokens 4096, identity jinja template, no start token, stop token 248044, the snapshot's tokenizer.json, prefer_activation_type fp32 on the decoder section; then add_executor_metadata.py for the 48 state buffers, 36 linear-attention + 12 K/V) (REPRODUCE.md rounds 3-4)` |
| **Quantization** | fp16 casting (weight-only FLOAT_CASTING, 16-bit, CHANNELWISE) of every FULLY_CONNECTED and EMBEDDING_LOOKUP weight, float compute (recipe wfp16); vision encoder and adapter fp16 float casting on FULLY_CONNECTED + CONV_2D; fp32 activation preference declared on the decoder section (Hub card Files; REPRODUCE.md round 4 table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `decider-2b-vision_fp16.litertlm` | `75b226796c43405b900399551487903dbe4f87d1a2c40aea422b2a9c33c8f62a` | 5254.132 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-09-29/decider-2b-vision-fp16__mac-studio-m4-max.json`, `data/device_runs/2.2.0/2026-09-29/decider-2b-vision-fp16__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 150 | - | 131.83 | - | 1675.9 | 53672.07 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-09-29 | measured |
| mac-studio-m4-max | cpu_xnnpack | pass | - | - | sig:reference readout decide() (decider_litert.py; CompiledModel CPU, 8 threads; warm caches) | - | - | - | - | 22101.25 | Mac Studio (M4 Max) · Apple M4 Max · litert 2.2.0 · macOS 27.0 (26A428) | 2026-09-29 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 150 | - | 825.94 | - | 291.2 | 15644.9 | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-09-29 | measured |

## Pitfalls

- The model returns option probabilities, not text: upstream reads the letter logits A-J at one answer slot per question (a ' (' token after ':' with 'Answer' within the 5 tokens before it), cuts them to the option count and softmaxes at T = 1. Through LiteRT-LM only the answer letter is available: the greedy first token of one question per request, one image first, and it is the top token over the whole vocabulary, not only over the option letters (on one two-question published row the top token for a two-option question is 'C'). LiteRT-LM's text-scoring call cannot stand in for the probabilities (its scores disagree with its own greedy decode on 0.17.1, LiteRT-LM #3561). Probabilities, text-only requests and several questions per request go through the bundled reference readout, reference/decider_litert.py (ai-edge-litert 2.2.0, numpy, Pillow, tokenizers; no torch, transformers or litert-lm) (Hub card Use it, Limitations).
- Fixed 256x256 input (64 image tokens) where upstream picks a resolution per image: for 224x224, 256x240 and 256x256 originals upstream's own processing gives the same pixels (13 published requests, 21 slots, |dp| 0); every other size gets fewer image tokens than upstream would use (70 for the 160x210 frames, 144 to 475 for the larger synthetic images), and upstream fp32 itself then matches its original-image answer on 30/32 published slots, max |dp| 0.373, p95 0.318, up to 0.58 on photographs and documents kept out of the repo. Resize to 256x256 with PIL bicubic before sending; the runtime's own image resize was never exercised (Hub card The price of the fixed 256x256 input, Limitations; FINDINGS section 2).
- Positions: the checkpoint uses 3-channel M-RoPE (image tokens on an 8x8 grid) and LiteRT-LM passes one 1-D position, so the decoder derives the three channels inside its graph with element-wise ops; plain 1-D positions flipped 2 of the 53 published image answers and moved one probability by 0.198 in upstream fp32. Text-only requests must start at position 65, where the derived positions keep upstream's relative positions; the runtime numbers them from 0 (the 9 text-only questions moved by up to 0.0497 with one flip in the fp32 graph), so text-only requests go through the reference readout (Hub card What matches upstream 4-5; FINDINGS sections 1 and 7).
- Rebuilders: do not export the derived rotary's step as clamp(x, 0, 1). The converter lowers it to RELU_0_TO_1 (9 per signature), which the WebGPU delegate of LiteRT-LM 0.17.1 on the Mac refuses ('RELU_0_TO_1: Not supported op', 25,675 of 25,927 prefill_1024 ops on the GPU, 'Hint fully delegated to single delegate is set, but the graph is not fully delegated', engine creation fails, twice). The shipped decoders write it as relu(x) - relu(x - 1) (equal on integer inputs; fp32 letter logits bit-equal) and create the full-GPU engine; DEQUANTIZE was listed in the same refusal but did not block delegation once RELU_0_TO_1 was gone (FINDINGS section 4).
- Agreement with upstream fp32 on the 42 published requests (53 image and 9 text-only answer slots; |dp| = the largest absolute probability difference over a slot's options): CPU exact-fit prefill 53/53 argmax, max |dp| 4.06e-05, p95 3.04e-05, text-only 9/9, max 2.41e-06; Mac GPU (Metal CompiledModel, one padded chunk per slot) 53/53, 4.07e-05, p95 3.31e-05, text-only 9/9, 1.10e-06. The reference's letter logits are bit-identical to the conversion's own graph readout on all 62 slots. Through LiteRT-LM 0.17.1 on the Mac, CPU and GPU gave upstream's letter as the first token on 12/12 single-question image rows, with prefill counts equal to upstream's and no 'Validation error' lines (Hub card Agreement with upstream; FINDINGS sections 5, 8, 10).
- Desktop file, closest to upstream; not for phones: XNNPACK unpacks fp16 weights to fp32 size, so LiteRT-LM's CPU cache folder holds 7,540,052,280 B for this decoder plus 1,312,687,576 B for the vision encoder and adapter, and it was not tried on the Galaxy S26. With the GPU backend the decoder program cache in a cache folder grew by 270,663,680 B at every engine creation in the lane's runs (821,634,912 B after three), and a warm folder did not shorten GPU engine creation (Hub card Files; FINDINGS section 6 and Corrections).
- Mac (Apple M4 Max, LiteRT-LM 0.17.1 PyPI, one decision per fresh process, 150 prompt tokens including 64 image tokens, medians of 3): CPU (8 threads) TTFT 1.68 s, engine creation 16.8 s, prefill 132 tok/s, peak RSS 53.7 GB and peak footprint 47.9 GB with cache_dir ':nocache' (1.72 s / 13.7 s / 115 tok/s / 21.2 / 6.6 GB with a warm cache folder; a warm-folder TTFT of 4.4-7.6 s appears when about 0.46-0.56 million page faults read the cache back from disk); GPU (WebGPU delegate over Metal) 0.29 s / 28.5 s / 826 tok/s / 15.6 / 19.3 GB (0.32 / 29.2 / 811 / 15.8 / 19.6 warm). The runtime's init_time_in_second reads about twice the Engine constructor wall; the card quotes the constructor wall (Hub card Performance; FINDINGS section 11).
- Reference readout on the same Mac (ai-edge-litert 2.2.0 CompiledModel CPU, 8 threads, section folder and weight cache already on disk): constructor 12.8 s, first decision in a process 1.28 s, second decision 0.90 s (the steady cost), peak RSS 22.1 GB; its first run copies the bundle's sections into .cache/readout/<bundle sha256>/ and writes the decoder's CPU weight cache there, 13,065,400,405 B in all for this file (Hub card Use it, Performance).
- Two different GPU paths on one Mac: LiteRT-LM's PyPI build runs Backend.GPU() through its statically linked WebGPU delegate (accelerator 'GPU WebGPU', adapter Apple M4 Max metal-3), while the reference readout's GPU rule uses ai-edge-litert 2.2.0 CompiledModel with the Metal accelerator (GpuOptions(enforce_f32=True), one padded prefill chunk per answer slot, slots within the first 1024 tokens); the Mac GPU probability numbers come from the second, the runtime rows from the first, and one does not stand in for the other (FINDINGS section 9; Hub card Use it).
- Bundle metadata shared by all three files: identity template (no role markers), no start token, stop token 248044, one 256x256 image, a 4096-token cache (rows longer than 4096 tokens do not fit; prompts above 2048 tokens were not exercised, and the derived rotary computes positions in float), ExecutorMetadata for the 48 state buffers and an fp32 activation preference for the decoder (no runtime log line states the precision the GPU executor used). One image per request, English only; the text behaviour is upstream's (decider-2b v5 text weights) (Hub card Files, Limitations; FINDINGS section 8 and Open questions).
- License Apache-2.0 as declared by the upstream checkpoint (source revision 863e290863655f1d6b69324d77d09ac972d21609); changes: text decoder, vision encoder and merger converted to LiteRT flatbuffers, M-RoPE derived in the decoder graph from a 1-D position, image fixed at 256x256, weights cast as listed, chat template replaced with an identity template, tokenizer repackaged unchanged, ExecutorMetadata and an fp32 activation preference added; the multi-token-prediction head is not included; community conversion, not an official Mapika or Qwen release (Hub card License and changes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
