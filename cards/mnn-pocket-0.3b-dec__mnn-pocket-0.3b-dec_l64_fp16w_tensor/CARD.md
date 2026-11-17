---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l64_fp16w_tensor
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT
task: text-classification
---

# mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l64_fp16w_tensor

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
| **Quantization** | float16 linear-layer weights, the LoRA adapter folded into the weights first; each norm computed on its input divided by 4 times its mean absolute value (README L20, L131) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `mnn-pocket-0.3b-dec_L64_fp16w_tensor.tflite` | `649e80b81cb07bad6f0f4cf209bddaefea3d305597d0926a9ab8450da861fd47` | 466.323 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.3.0/2026-11-08/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l64_fp16w_tensor__mac-studio-m4-max.json`, `data/device_runs/2.3.0/2026-11-08/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l64_fp16w_tensor__pixel-11-pro.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu_xnnpack | fallback | no | - | sig:the line: 71 questions (the requests that fit L64) against the source model's reference; Python CompiledModel (ai-edge-litert 2.3.0), CPU XNNPACK, 4 threads | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert 2.3.0 · macOS 27.0 (26A428) | 2026-11-08 | measured |
| mac-studio-m4-max | gpu_metal_mac | pass | yes | - | sig:the line: 71 questions (the requests that fit L64) against the source model's reference; Python CompiledModel (ai-edge-litert 2.3.0), GPU Metal, GpuOptions(enforce_f32=True) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert 2.3.0 · macOS 27.0 (26A428) | 2026-11-08 | measured |
| pixel-11-pro | npu_google_tensor | pass | yes | - | sig:the line: 71 questions of the L64 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0) with --npu-vendor google_tensor, Google Tensor NPU, the graph compiled on the phone | - | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert 2.3.0 · LiteRT Google Tensor NPU runtime 2.3.0 (Maven litert-npu-runtime-google-tensor:2.3.0; compiler plugin 6b1432e7…, dispatch e3814400…) · Android 17 (CD1A.260905.001.B1) | 2026-11-08 | measured |

## Pitfalls

- A decision model, not a chat model: it takes a `state` and typed `questions` (`choice`, `noul`, `score`) and returns the probability of every candidate in one forward pass; it never generates text (README L18).
- The graph is the model body only: it takes the embeddings of the request's tokens, the rotary tables and the attention mask and returns the hidden states after the final norm; the Python host in `host/` builds the request, takes the embeddings from `embedding.safetensors` (the source model's token table) and reads the answers with the source model's head and temperature, as the source model's `inference.py` does (README L20, L147-L148).
- Each file is a static window: `<n>` in the name is the longest request it takes; run the smallest file that holds the request. A request longer than 2,048 tokens raises `ValueError` in the host, which never cuts it; the source model's own limit is 32,768 tokens (README L71, L84).
- The checks measure agreement with the source model's `inference.py`, unchanged, on its own files (float32, CPU, one thread): 2,409 questions on the Mac, a fixed subset of 16 to 322 questions per file on the phones; the bar is the same winner on every question whose two most likely options are more than 0.02 apart, a largest |dp| of at most 0.02 and a mean of at most 0.002 (README L88, L102).
- The repository has no Android host yet: an Android app has to port `host/pocket_dec_litert.py`; the phones ran LiteRT 2.3.0 through its C API with inputs the Mac computed with the host's code (README L137).
- Runs through LiteRT's Google Tensor NPU runtime (Maven `litert-npu-runtime-google-tensor` 2.3.0) on the Pixel 11 Pro (Google Tensor G6), each file in one partition with all its ops on the NPU; LiteRT compiles it on the phone at the first load (README L82, L131).
- Each norm is computed on its input divided by 4 times its mean absolute value, which gives the same result in exact arithmetic: this NPU ran the graphs in FP16, where values below 2^-14 became 0, and in the other files the squares of small values in the norms were lost (README L131).
- 64 to 256 tokens only: no 512-token Tensor file was compiled; longer requests run on the Pixel 11 Pro's GPU at FP32 (README L82, L131).
- Passes the bar on all four files (the same winner on every question, near ties included; largest |dp| 0.0110, mean at most 0.0017) and answers the source model's default example `shipping`, as the source model does (README L100, L104).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
