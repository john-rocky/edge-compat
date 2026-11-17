---
family: wevisone-vl
license: apache-2.0
model_id: wevisone-vl-4b-int8
source_url: https://huggingface.co/litert-community/WeVisOne-VL-4B
task: image-text-to-text
---

# wevisone-vl-4b-int8

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | wevisone-vl |
| **Source** | https://huggingface.co/litert-community/WeVisOne-VL-4B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (L22); the same export as the repo's int4 file — each image token carries the main embedding and Qwen3-VL's three DeepStack features, the decoder computes Qwen3-VL's positions in the graph, the vision tower is rewritten in plain PyTorch for a fixed 512 x 512 input without a gather op (L153-L155) — with the int8 recipe below and without the activation declaration and the GPU graph step (L157); since 2026-11-12 the decoder's and the token embedder's weights sit in TFLiteWeights sections at the end of the file (LiteRT's external weights) and the vision encoder and adapter keep theirs in the graph (L33, L164); the int8 file has no REPRODUCE entry (L160) litert-torch 0.10.0 (L22), ai-edge-quantizer 0.10.0 (L157); the phone numbers on LiteRT-LM 0.19.1's litert_lm_advanced_main built from the tag for android_arm64 (L44, L95); the Mac table on the PyPI litert-lm 0.19.1 package (L103) |
| **Command** | `tencent/WeVisOne-VL-4B float32 release rev 4a7bed92 (its bfloat16 release 6b44a458 holds the same weights) -> the WeVisOne-VL-2B recipe of hf-to-litertlm (L160) -> the float32 export quantized with ai-edge-quantizer 0.10.0's dynamic int8 recipe on the decoder, its output layer and the token embedder (L157); a single-picture fast_vlm file with 2,048 tokens of context for the prompt and the answer (L33); re-packed with LiteRT external weights on 2026-11-12, the same weight bytes (L164)` |
| **Quantization** | decoder, its output layer and the token embedder int8 with one scale per output channel; vision encoder and adapter float16; no activation type declared and no GPU graph step: the file for a CPU (L31, L33, L157) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `WeVisOne-VL-4B_int8.litertlm` | `c288209eb368095d71bc26e9d75d88b42252000b886c02fc2cd0d773679a9adf` | 5110.785 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-10/wevisone-vl-4b-int8__galaxy-s26.json`, `data/device_runs/0.19.1/2026-11-12/wevisone-vl-4b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | pass | sig:the suite: 29 picture questions (17 pictures, 512 x 512 PNG) + 10 text questions, one process per question, greedy | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| galaxy-s26 | cpu | pass | - | - | 297 | - | 106.97 | 10.25 | 4486.6 | 7041.98 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| galaxy-s26 | cpu | pass | - | - | 1024 | - | 213.12 | 8.24 | 4920.0 | 5614.76 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 297 | - | 36.43 | 24.79 | 8699.8 | 27178.5 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 434.26 | 23.92 | 2399.9 | 25830.5 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |

## Pitfalls

- External weights (since 2026-11-12): on an Android GPU the file needs LiteRT-LM 0.19.0 or later; the GPU engine of the v0.18.0 and v0.18.1 Android builds crashes (SIGSEGV) while it is being built (LiteRT-LM#3986). For those builds the earlier file at revision ee03d010 keeps every weight inside the graph; the new file holds the same weight bytes and gave the same answers as the earlier one to 15 questions on the Galaxy S26 CPU (LiteRT-LM 0.19.1) and to 3 on the Mac GPU (L33, L164).
- Which bytes the numbers are: the card's Mac table ran the file of 2026-11-12; every other measurement on the card (2026-10-27 to 2026-11-11) ran the earlier file, and on the phone speed, engine creation and memory were not measured again (L164).
- For a CPU: the card's runs use this file on the CPU with --backend=cpu --vision_backend=cpu --num_cpu_threads=4 and the repo's int4 file on the GPU (L33, L52). On the Galaxy S26 GPU (LiteRT-LM 0.18.1) the test harness stopped a run of this file when the free memory fell to 0.71 GB (L146).
- One picture size: the file always reads 512 x 512; turn the picture upright from its EXIF orientation and stretch it with Pillow's BICUBIC filter, no padding or cropping, saved as PNG; the coordinates then map back onto the upright picture by its width and height (L37, L143).
- Text-only conversations: without a picture, every token after the 271st of the conversation (prompt, answer and chat template together) gets a wrong position, and from there the answer can differ from the original's (L144).
- Later messages: from the second message on, LiteRT-LM passes the previous answer's stop token to the model with a zero embedding, so the second answer starts from a context the original never sees; ask one question per conversation (L73, L145).
- Segmentation and matting: the file does not contain the source model's 9.6M-parameter pixel decoder; to those templates it answers with a box that ends in an <OBJ> token, and no mask (L26, L69).
- Absent objects: to a referring question with no matching object the file answered <box>none</box>; asked to count airplanes with boxes in a radar picture that holds none, it answered one small box and 'Total number: 1.' — as the original does both times (L69).
- One OCR difference from the original: on 1 of the 29 picture questions, on the Galaxy S26's CPU, the file leaves out 'Image', a small label in the picture's corner that the original reads first, in each of three separate processes, and reads the other five words as the original does; this puts its OCR error at 0.098 there, above the 0.05 that the card's agreement rule allows. On the 21 questions with a known answer it scores as the original, the same answers on LiteRT-LM 0.19.1 and 0.18.1 (L119, L121, L127, L129).
- Startup on the Galaxy S26 CPU (build A): the first process wrote about 5.69 GB of cache files next to the model and took 4.9 s of engine creation; the later ones 0.39 s, then another 0.61 s to open the picture conversation (L52, L99, L101).
- Speeds with the phone's frequency limit: each process started with no limit on the CPU's frequency, and the phone set one during each run, so the speeds include that slowdown (L101).
- Memory: the 12 GB Galaxy S26 ran 91 processes of this file on its CPU (LiteRT-LM 0.18.1 and 0.19.1) and stopped none; the highest peak memory was 8.00 GB, the picture question's 7.26-7.48 GB; phones with less memory were not tried (L99, L146).
- On a Mac CPU (Apple M4 Max, the PyPI litert-lm 0.19.1 package, the decoder with the runtime's default 4 threads, no cache files): engine creation 15.17 s, the picture's first token 8.70 s, decode 24.79 tok/s, text prefill 434.26 tok/s, and a peak memory of 28.50 GB in the picture processes (L103, L109, L111).
- Android program: the v0.19.1 release has no Android binary; the card's runs use litert_lm_advanced_main built from the tag with the tag's prebuilt/android_arm64 libraries beside it and LD_LIBRARY_PATH set to them, one process per answer, the prompt file holding [image:<path>] and then the question with nothing between them and no newline at the end (L44, L52).
- Sampling: the file carries no sampler settings, so LiteRT-LM 0.19.1 decodes it greedily; the source package's repetition penalty of 1.05 was not tested (L50, L147).
- Not tested: this file in an app (on the phone it ran through litert_lm_advanced_main only), the files of 2026-11-12 in an app or on Android with LiteRT-LM 0.18.x, the known-answer and agreement tests on a desktop with LiteRT-LM 0.19.1, sampled answers; there is no NPU build (L147).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
