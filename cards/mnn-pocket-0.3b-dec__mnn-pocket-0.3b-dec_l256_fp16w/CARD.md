---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT
task: text-classification
---

# mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w

| | |
|---|---|
| **Task** | text-classification |
| **Family** | mnn-pocket |
| **Source** | https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (for LiteRT 2.3.0) |
| **Command** | `the repository's conversion/ (conversion/README.md) holds the scripts and REPRODUCE.md every check (README L150); the card names the tools, not a command line` |
| **Quantization** | float16 linear-layer weights, the LoRA adapter folded into the weights first (README L20) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `mnn-pocket-0.3b-dec_L256_fp16w.tflite` | `7fb2e0c29c7d93de1f6a51c919711de2a5aa2aa42fd6a9c31ffc96072ce4f92a` | 466.233 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.3.0/2026-11-07/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w__galaxy-s26.json`, `data/device_runs/2.3.0/2026-11-07/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w__mac-studio-m4-max.json`, `data/device_runs/2.3.0/2026-11-07/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w__pixel-11-pro.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | fallback | no | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), CPU XNNPACK, 1 thread (this build passed the thread count under an option name LiteRT 2.3.0 does not read) | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), GPU OpenCL, GpuOptions precision FP16_WITH_FP32_ACCUM | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), GPU OpenCL, GpuOptions precision FP32 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |
| mac-studio-m4-max | cpu_xnnpack | fallback | no | - | sig:the line: 379 questions (the requests that fit L256) against the source model's reference; Python CompiledModel (ai-edge-litert 2.3.0), CPU XNNPACK, 4 threads | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert 2.3.0 · macOS 27.0 (26A428) | 2026-11-07 | measured |
| mac-studio-m4-max | gpu_metal_mac | pass | yes | - | sig:the line: 379 questions (the requests that fit L256) against the source model's reference; Python CompiledModel (ai-edge-litert 2.3.0), GPU Metal, GpuOptions(enforce_f32=True) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert 2.3.0 · macOS 27.0 (26A428) | 2026-11-07 | measured |
| pixel-11-pro | cpu_xnnpack | fallback | no | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), CPU XNNPACK, 1 thread (this build passed the thread count under an option name LiteRT 2.3.0 does not read) | - | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert 2.3.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-07 | measured |
| pixel-11-pro | gpu_mldrift | pass | yes | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), GPU OpenCL, GpuOptions precision FP32 | - | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert 2.3.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-07 | measured |

## Pitfalls

- A decision model, not a chat model: it takes a `state` and typed `questions` (`choice`, `noul`, `score`) and returns the probability of every candidate in one forward pass; it never generates text (README L18).
- The graph is the model body only: it takes the embeddings of the request's tokens, the rotary tables and the attention mask and returns the hidden states after the final norm; the Python host in `host/` builds the request, takes the embeddings from `embedding.safetensors` (the source model's token table) and reads the answers with the source model's head and temperature, as the source model's `inference.py` does (README L20, L147-L148).
- Each file is a static window: `<n>` in the name is the longest request it takes; run the smallest file that holds the request. A request longer than 2,048 tokens raises `ValueError` in the host, which never cuts it; the source model's own limit is 32,768 tokens (README L71, L84).
- The checks measure agreement with the source model's `inference.py`, unchanged, on its own files (float32, CPU, one thread): 2,409 questions on the Mac, a fixed subset of 16 to 322 questions per file on the phones; the bar is the same winner on every question whose two most likely options are more than 0.02 apart, a largest |dp| of at most 0.02 and a mean of at most 0.002 (README L88, L102).
- The repository has no Android host yet: an Android app has to port `host/pocket_dec_litert.py`; the phones ran LiteRT 2.3.0 through its C API with inputs the Mac computed with the host's code (README L137).
- On a GPU set float32 precision (`GpuOptions(enforce_f32=True)` in the Python host, `CompiledModel.GpuOptions.Precision.FP32` in Kotlin, `kLiteRtDelegatePrecisionFp32` in the C API): an app that sets no precision gets FP16 on both phones, and FP16 changes the answers: among questions whose top two options are more than 0.02 apart the winner changed on up to 5 of 305 on the Galaxy S26, 1 of 305 on the Pixel 11 Pro and 9 of 2,314 on Apple Metal at its default precision, and a probability moved by up to 0.068, 0.029 and 0.058 (README L73).
- FP16 with FP32 accumulation passes the bar on the Galaxy S26 (44 of 54 near ties keep the reference's winner, largest |dp| 0.0126) and misses the mean on 3 of the 6 files it ran on the Pixel 11 Pro; it can flip a near tie: in the source model's default example (`department`: `shipping` 0.41064 against `returns` 0.40709) it answers `returns` (README L96, L102, L104).
- Set the CPU thread count: with no CPU options LiteRT 2.3.0's C API ran on one thread, and the 128-token file took 1.3 times as long as at 4 threads on the Galaxy S26 and 2.2 times as long on the Pixel 11 Pro; the card's phone CPU rows use 4 threads (Galaxy S26) and 6 (Pixel 11 Pro), whose probabilities equal those of one thread to every digit (README L133).
- LiteRT's GPU cache (`serializationDir`, `modelCacheKey`, `serializeProgramCache` and `serializeExternalTensors` in `CompiledModel.GpuOptions`) cut the Pixel 11 Pro's creation of the 128-token model from 15.5 s to 1.8 to 2.0 s, but each call then took 12 to 19 % longer; on the Galaxy S26 it cut 1.6 s to 1.3 s at the same speed; the cache takes 0.93 GiB on disk and raises the process's peak by about 0.9 GiB (README L135).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
