---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w_npu_sm8850
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT
task: text-classification
---

# mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w_npu_sm8850

| | |
|---|---|
| **Task** | text-classification |
| **Family** | mnn-pocket |
| **Source** | https://huggingface.co/litert-community/MNN-Pocket-0.3B-Dec-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (for LiteRT 2.3.0); compiled for the SM8850 with QAIRT 2.50 |
| **Command** | `the repository's conversion/ (conversion/README.md) holds the scripts and REPRODUCE.md every check (README L150); the card names the tools, not a command line` |
| **Quantization** | float16 linear-layer weights, the LoRA adapter folded into the weights first; SiLU written with EXP in place of LOGISTIC for the NPU; compiled for the SM8850 ahead of time (README L20, L80, L129) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `mnn-pocket-0.3b-dec_L256_fp16w_npu_SM8850.tflite` | `b5c25051c96cc5f396b1f017bc4823afc9c2df0353687ca609218c1589d3350a` | 475.242 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.3.0/2026-11-07/mnn-pocket-0.3b-dec__mnn-pocket-0.3b-dec_l256_fp16w_npu_sm8850__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | npu_qnn | pass | yes | - | sig:the line: 61 questions of the L256 set against the source model's reference; the C-API runner build 7e7539e5 (LiteRT 2.3.0), Qualcomm HTP, the SM8850 context compiled ahead of time on the Mac, HTP performance mode Burst | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · QAIRT 2.50.0.260828221209 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |
| galaxy-s26 | npu_qnn | pass | yes | - | sig:timing: `default_request` (inference.py's DEFAULT_REQUEST) (85 tokens, 3 questions), the median of 20 calls after 5 warm-up calls in one process, a call = write the four inputs, run, read `hidden` back; the C-API runner build 7e7539e5 (LiteRT 2.3.0), Qualcomm HTP, the SM8850 context compiled ahead of time on the Mac, HTP performance mode Burst | 39.4925 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · QAIRT 2.50.0.260828221209 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |
| galaxy-s26 | npu_qnn | pass | yes | - | sig:timing: `readme_request` (the source model card's example) (60 tokens, 1 question), the median of 20 calls after 5 warm-up calls in one process, a call = write the four inputs, run, read `hidden` back; the C-API runner build 7e7539e5 (LiteRT 2.3.0), Qualcomm HTP, the SM8850 context compiled ahead of time on the Mac, HTP performance mode Burst | 39.579499999999996 | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert 2.3.0 · QAIRT 2.50.0.260828221209 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-11-07 | measured |

## Pitfalls

- A decision model, not a chat model: it takes a `state` and typed `questions` (`choice`, `noul`, `score`) and returns the probability of every candidate in one forward pass; it never generates text (README L18).
- The graph is the model body only: it takes the embeddings of the request's tokens, the rotary tables and the attention mask and returns the hidden states after the final norm; the Python host in `host/` builds the request, takes the embeddings from `embedding.safetensors` (the source model's token table) and reads the answers with the source model's head and temperature, as the source model's `inference.py` does (README L20, L147-L148).
- Each file is a static window: `<n>` in the name is the longest request it takes; run the smallest file that holds the request. A request longer than 2,048 tokens raises `ValueError` in the host, which never cuts it; the source model's own limit is 32,768 tokens (README L71, L84).
- The checks measure agreement with the source model's `inference.py`, unchanged, on its own files (float32, CPU, one thread): 2,409 questions on the Mac, a fixed subset of 16 to 322 questions per file on the phones; the bar is the same winner on every question whose two most likely options are more than 0.02 apart, a largest |dp| of at most 0.02 and a mean of at most 0.002 (README L88, L102).
- The repository has no Android host yet: an Android app has to port `host/pocket_dec_litert.py`; the phones ran LiteRT 2.3.0 through its C API with inputs the Mac computed with the host's code (README L137).
- Set the HTP performance mode to Burst: without it the calls on the Galaxy S26 took 5.5 to 6.7 times as long (76.3 ms against 11.3 ms for the 60-token example); in the C API it is `htp_performance_mode = 2` (`kLiteRtQualcommHtpPerformanceModeBurst`) in the Qualcomm options, in Kotlin `CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST)` set as `qualcommOptions` of `CompiledModel.Options` (README L127).
- Compiled ahead of time for the Qualcomm SM8850 (Galaxy S26) with QAIRT 2.50: on the Galaxy S26 a file loads in 0.2 s (64 tokens) to 0.5 s (1,024 tokens) and 1.2 s (2,048 tokens), and the process peaks at 0.55 to 0.70 GB; for a Qualcomm NPU compiled on the phone the repository has the `_npu` files (README L80-L81, L129).
- SiLU is written with EXP in place of LOGISTIC, because the NPU's LOGISTIC returned 0 for inputs below about -5.7 (measured on the Galaxy S26 with QAIRT 2.47, on another model's graph) (README L129).
- Running it needs LiteRT's Qualcomm NPU runtime (Maven `litert-npu-runtime-qualcomm` 2.3.0) and Qualcomm's QAIRT 2.50 libraries; every file ran on the NPU in one partition, with all 2,600 ops (README L129).
- The Qualcomm NPU passes the bar (every winner that is not a near tie, 42 of 54 near ties, largest |dp| 0.0190) but can flip a near tie: in the source model's default example (`department`: `shipping` 0.41064 against `returns` 0.40709) it answers `returns`; from 64 to 1,024 tokens the files compiled ahead of time and the files compiled on the phone gave the same probabilities to every digit (README L97, L102, L104).
- The repository's scripts rebuild the 20 graphs other than the SM8850 files byte for byte on the Mac that made them; compiled again, the 64- and 768-token SM8850 files were the same computation, not the same bytes (README L150).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
