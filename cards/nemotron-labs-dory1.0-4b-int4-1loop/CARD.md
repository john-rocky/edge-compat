---
family: dory
license: other (openmdw-1.1)
model_id: nemotron-labs-dory1.0-4b-int4-1loop
source_url: https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B
task: text-generation
---

# nemotron-labs-dory1.0-4b-int4-1loop

| | |
|---|---|
| **Task** | text-generation |
| **Family** | dory |
| **Source** | https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B |
| **License** | other (openmdw-1.1) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 `export_hf` with a model definition of its own (transformers has no Dory class; the formulas follow NVIDIA's vLLM pull request); repro = REPRODUCE.md "NVIDIA-Nemotron-Labs-Dory1.0-4B" (card L24, L130, L136) litert-torch 0.9.4, ai-edge-quantizer 0.9.0, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1; packed with litert-lm-builder 0.18.0 (card L130) |
| **Command** | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 rev b366b87f and this repository's int4 file (both downloaded at pinned commits and checked) -> the graph exported again with n_recurrent_loops set to 1 in the config (the same arguments otherwise: prefill signatures 128 / 512 / 1024 plus decode over a 2048-token KV budget, 36 KV caches, the MLP normalization rewrite of the same 11 blocks) -> data-free int4 with the int4 recipe -> the int4 file's GPTQ codes written weight by weight (each weight found by its block scales, which the data-free quantization reproduces; 289 of 289) -> GPU composites (426 in its four graphs) -> packed with the int4 file's metadata and embedding sections; run twice on 2026-10-23, each in a fresh directory, the script rebuilt every section of the file byte for byte (card L136)` |
| **Quantization** | the int4 file's weights byte for byte: int4 blockwise-32 on the linear layers with the int4 file's GPTQ codes and block scales; the int4 file's int8 embedding section; GPU composites; no activation type declared (the GPU runs the runtime's default fp16 activations) (card L32, L34) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `NVIDIA-Nemotron-Labs-Dory1.0-4B_int4_1loop.litertlm` | `53f935834360de82ba553b630e124578fb1384c6ff59675f035ae7023450dab4` | 2289.351 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-10-23/nemotron-labs-dory1.0-4b-int4-1loop__galaxy-s26.json`, `data/device_runs/0.16.1/2026-10-24/nemotron-labs-dory1.0-4b-int4-1loop__galaxy-s26.json`, `data/device_runs/0.16.1/2026-10-26/nemotron-labs-dory1.0-4b-int4-1loop__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-23/nemotron-labs-dory1.0-4b-int4-1loop__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-23/nemotron-labs-dory1.0-4b-int4-1loop__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-54 (GSM8K test question 54, answer 40), thinking off, the one-loop file then the int4 file, screen recording off | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-24 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-54 (GSM8K test question 54, answer 40), thinking off, the one-loop file then the int4 file, screen recording on | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-24 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-75 (GSM8K test question 75, answer 60), thinking off, gsm8k-75 alone, picked from the Examples list with 'Both' (the one-loop file, then the int4 file), run t1, screen recording on | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-26 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-75 (GSM8K test question 75, answer 60), thinking off, gsm8k-75 alone, picked from the Examples list with 'Both' (the one-loop file, then the int4 file), run u2, screen recording off | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-26 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-75 (GSM8K test question 75, answer 60), thinking off, the sample's eight bundled examples with the int4 file first (this row = gsm8k-75, the sixth), run b2, screen recording off | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-26 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android app (hfmodels-android samples/loops, debug build); gsm8k-75 (GSM8K test question 75, answer 60), thinking off, the sample's eight bundled examples with the one-loop file first (this row = gsm8k-75, the sixth), run b1, screen recording off | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-26 | measured |
| galaxy-s26 | gpu | pass | yes | - | sig:Android library (hfmodels-android samples/chat ChatDeviceCheck, debug build); one conversation, 'What is 17 + 25? Answer briefly.' (expect '42') | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-23 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 439.6025 | 17.3225 | 2387.5 | 1552.286 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-23 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1619.61 | 88.175 | 643.6 | 3668.971 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-23 | measured |

## Pitfalls

- One loop instead of two: the same int4 weights with the shared block of layers applied once per token; on GSM8K (questions 0-99, thinking off, Mac GPU) 86 / 100 against the int4 file's 90, the two files disagreeing on 10 questions (7 right with two loops only, 3 with one loop only); the BF16 reference shows the trade too: 86 with one loop against 95 with two, 10 questions right with two loops only and 1 with one loop only, and on question 75 it also answers wrong with one loop and right with two; NVIDIA recommends two loops and the card names the int4 file as the one to start with (card L36, L63, L81, L83, L85).
- Speed: in runs that alternated the two int4 files (2026-10-23), decode 88.18 against 58.46 tok/s on the Mac M4 Max GPU (x 1.51) and 17.32 against 11.08 on the Galaxy S26 GPU (x 1.56), prefill x 1.57 / x 1.60, and 4.52 against 5.38 GB of GPU memory on the Galaxy S26 (card L78-L80).
- Near-tie answers move with the device: asked the rhyme question ('Roses are red, violets are ___'), the file answered 'red' on the Mac GPU (with fp32 activations too), 'purple' on the Mac CPU and 'blue' on the Galaxy S26 GPU; the PyTorch reference with the same one-loop int4 weights answers 'blue', and along the Mac GPU's fp32-activation text it picks the same next token at 932 of 935 steps, the first difference at the 39th token of the thought where its own two best tokens were 0.016 apart (card L52, L83).
- GPU only: the card measures the file on GPUs (Mac WebGPU on Metal, the Galaxy S26 OpenCL delegate with every node of the four graphs on it); the Pixel 8a row reads not run, and the int8 file is the CPU file (card L36, L42, L52).
- fp16 activations: the same MLP normalization rewrite covers the one-loop graph: in an fp16 emulation over the same 25 prompts no row overflows with the rewrite, and without it every overflowing row is in one of the 11 rewritten blocks (card L136).
- Phone heat: both Galaxy S26 GPU runs of the file (2026-10-23) started cool (thermal status 0) and ended at status 2, the GPU clock under its cap in 19 of 29 samples each (card L122).
- Thinking and template: thinking is on by default (the generation prompt ends with <think> and a newline, the bundle declares the thought channel); rendered with enable_thinking false the prompt ends with <think></think>; NVIDIA recommends temperature 1.0 and top-p 0.95, the bundle sets no sampler; stop tokens </s> and <|im_end|> plus 14 stop sequences; model type generic_model (tool calls not parsed; tool calling not tested) (card L97).
- Final mark on LiteRT-LM 0.16.1 and earlier: of the 14 stop sequences, 13 put one of 13 characters (12 punctuation marks and a space) in front of <|im_end|> and a newline; an answer whose last mark is a token of its own, such as a ? or . after a word, arrives without that mark on LiteRT-LM 0.16.1 (the runtime of the Android app under Loops) and on 0.11.0 to 0.16.0 (source read): the runtime holds the mark back, the answer ends on <|im_end|> alone and the held mark is never written; a mark the tokenizer merges with the text before it (**., ).) is kept; 0.17.0 and later write the held mark (card L97).
- Second turn with thinking off: from the second turn of a conversation on, LiteRT-LM puts the previous answer's stop token into the KV cache with a zero embedding, because the file holds the embedding table in its own section and the decoder takes embeddings (read in the source of 0.16.0 to 0.18.1; measured with 0.18.1 on WeVisOne-VL-2B and Qwen2.5-Coder-1.5B-Instruct, not on these files), so later answers can differ from the answer to the same messages in one prompt; with thinking on the runtime rebuilds the cache at each turn, and the source shows a zero embedding at another token from the third turn on (not measured); one question per conversation, or a new conversation that starts with the earlier messages and the new question, avoids both (card L101).
- Conversation preface with tools and no messages: the template reads messages[0] without checking that the list is non-empty, so LiteRT-LM's render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages; read in the source of 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on these files' template text; the default path, where the first message is rendered with the preface, is not affected; a system message or any initial message in the conversation config avoids it (card L141).
- Cache: without --cache no, the CLI can write a compiled cache file next to the model on its first run; without --backend, `run` uses the CPU (card L95).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
