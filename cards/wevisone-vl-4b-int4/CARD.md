---
family: wevisone-vl
license: apache-2.0
model_id: wevisone-vl-4b-int4
source_url: https://huggingface.co/litert-community/WeVisOne-VL-4B
task: image-text-to-text
---

# wevisone-vl-4b-int4

| | |
|---|---|
| **Task** | image-text-to-text |
| **Family** | wevisone-vl |
| **Source** | https://huggingface.co/litert-community/WeVisOne-VL-4B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (L22); the decoder carries Qwen3-VL's three DeepStack features in each image token's row and computes the three-axis positions in the graph, the vision tower is rewritten in plain PyTorch for a fixed 512 x 512 input without a gather op, and after the export hf-to-litertlm's gpu_graph_retrofit.py rewrote the decoder's KV-cache writes and attention products into LiteRT-LM's GPU composites (odml.cache_update, odml.runtime_bmm); one fast_vlm bundle with the source repository's own tokenizer.json byte for byte (L33, L153-L158); since 2026-11-12 the decoder's and the token embedder's weights sit in TFLiteWeights sections at the end of the file (LiteRT's external weights) and the vision encoder and adapter keep theirs in the graph (L33, L164); recipe = hf-to-litertlm REPRODUCE.md's WeVisOne-VL-4B entry, which rebuilds the earlier int4 file (revision ee03d010, before its weights moved) with every section equal to that file's (L160) litert-torch 0.10.0 (L22), ai-edge-quantizer 0.10.0 (L156); the phone numbers on LiteRT-LM 0.19.1's litert_lm_advanced_main built from the tag for android_arm64 (L44, L77); the Mac table on the PyPI litert-lm 0.19.1 package (L103) |
| **Command** | `tencent/WeVisOne-VL-4B float32 release rev 4a7bed92 (its bfloat16 release 6b44a458 holds the same weights) -> the WeVisOne-VL-2B recipe of hf-to-litertlm at the 4B's dimensions, with the int4 decoder and the GPU graph step (L160); a single-picture fast_vlm file with 2,048 tokens of context for the prompt and the answer (L33); re-packed with LiteRT external weights on 2026-11-12, the same weight bytes (L164)` |
| **Quantization** | decoder int4 blockwise 32 with one scale per block, every block scale at or above 2^-14, its output layer included; token embedder int8 (one scale per output channel); vision encoder and adapter float16; float32 activations declared for the decoder (L30, L156) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `WeVisOne-VL-4B_int4.litertlm` | `6ae2e63e83516bf8a8f382c9938c0be8f716cba2a66b8bbcc22991e9123c1e48` | 3345.455 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.1/2026-10-27/wevisone-vl-4b-int4__mac-studio-m4-max.json`, `data/device_runs/0.19.1/2026-11-10/wevisone-vl-4b-int4__galaxy-s26.json`, `data/device_runs/0.19.1/2026-11-12/wevisone-vl-4b-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | - | pass | sig:the suite: 29 picture questions (17 pictures, 512 x 512 PNG) + 10 text questions, one process per question, greedy | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| galaxy-s26 | gpu | pass | - | - | 297 | - | 87.275 | 15.2 | 4864.2 | 3608.69 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| galaxy-s26 | gpu | pass | - | - | 1024 | - | 108.195 | 15.23 | 9545.0 | 1870.16 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-10 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 297 | - | 61.2 | 23.68 | 5393.5 | 17092.9 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 117.81 | 25.93 | 8730.7 | 15505.1 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the suite: 29 picture questions (17 pictures, 512 x 512 PNG) + 10 text questions, one process per question, greedy | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 297 | - | 876.47 | 92.34 | 485.1 | 6810.4 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1249.49 | 89.17 | 830.8 | 5092.0 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |

## Pitfalls

- External weights (since 2026-11-12): on an Android GPU the file needs LiteRT-LM 0.19.0 or later; the GPU engine of the v0.18.0 and v0.18.1 Android builds crashes (SIGSEGV) while it is being built (LiteRT-LM#3986). For those builds the earlier file at revision ee03d010 keeps every weight inside the graph; the new file holds the same weight bytes and gave the same answers as the earlier one to 15 questions on the Galaxy S26 GPU (LiteRT-LM 0.19.1) and to 3 on the Mac GPU (L33, L164).
- Which bytes the numbers are: the card's Mac table ran the file of 2026-11-12; every other measurement on the card (2026-10-27 to 2026-11-11) ran the earlier file, and on the phone speed, engine creation and memory were not measured again (L164).
- One picture size: the file always reads 512 x 512, and every comparison on the card is at that size; turn the picture upright from its EXIF orientation and stretch it with Pillow's BICUBIC filter, no padding or cropping, saved as PNG (other resizing not tested). The source package chooses the resolution per picture; how far the original's answers at its own resolution differ was not measured for the 4B (L37, L143).
- Text-only conversations: without a picture, every token after the 271st of the conversation (prompt, answer and chat template together) gets a wrong position, and from there the answer can differ from the original's; conversations with a picture are not affected (L144).
- Later messages: from the second message on, LiteRT-LM passes the previous answer's stop token to the model with a zero embedding, so the second answer starts from a context the original never sees; ask one question per conversation and send the picture again for the next (L73, L145).
- Two LiteRT-LM 0.19.1 builds for Android: the tag build links the GPU accelerator in from source. On the Galaxy S26, with this file, it prefilled the 1,024-token text benchmark faster (108.20 against 90.45 tok/s) but reached the picture question's first token later (4.86 against 4.22 s) than the same commit loading the prebuilt libLiteRtGpuAccelerator.so, in both pairs of an A B B A session; only the tag's default build had its answers checked (L79-L84, L165).
- Segmentation and matting: the file does not contain the source model's 9.6M-parameter pixel decoder; to those templates it answers with a box that ends in an <OBJ> token, and no mask (L26, L69).
- Absent objects: to a referring question with no matching object it answered <box>none</box>; asked to count airplanes with boxes in a radar picture that holds none, it answered one small box and 'Total number: 1.' — the original does the same both times (L69).
- Startup on the Galaxy S26 (build A): engine creation 11.5 s, then opening the picture conversation another 4.9 s, then the first token at 4.86 s; the first run writes about 0.46 GB of cache files next to the model (L50, L81, L84).
- Memory: on LiteRT-LM 0.19.1 the 12 GB Galaxy S26 ran 57 processes of this file and stopped none; the highest peak memory of any process was 4.02 GB, the picture question's 3.68-3.80 GB over the two builds; phones with less memory were not tried (L81-L82, L146).
- A CPU: use the repo's int8 file; this file's answers were not checked on a CPU. On a Mac CPU (Apple M4 Max, the PyPI litert-lm 0.19.1 package, the decoder with the runtime's default 4 threads, no cache files) it ran only speed rows: the picture's first token 5.39 s, decode 23.68 tok/s, peak memory 17.80-17.93 GB, against 0.49 s, 92.34 tok/s and 7.13-7.14 GB on that Mac's GPU (L33, L103-L111, L147).
- Android program: the v0.19.1 release has no Android binary; the card's runs use litert_lm_advanced_main built from the tag with the tag's prebuilt/android_arm64 libraries beside it and LD_LIBRARY_PATH set to them, one process per answer, the prompt file holding [image:<path>] and then the question with nothing between them and no newline at the end (L44-L47).
- Sampling: the file carries no sampler settings, so LiteRT-LM 0.19.1 decodes it greedily; the source package's repetition penalty of 1.05 was not tested (L50, L147).
- Not tested: the known-answer and agreement tests on a CPU or on a desktop with LiteRT-LM 0.19.1, the files of 2026-11-12 in an app (the card's one test app ran the earlier file) or on Android with LiteRT-LM 0.18.x, other apps, sampled answers; there is no NPU build (L54, L147).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
