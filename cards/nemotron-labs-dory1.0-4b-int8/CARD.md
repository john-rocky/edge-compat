---
family: dory
license: other (openmdw-1.1)
model_id: nemotron-labs-dory1.0-4b-int8
source_url: https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B
task: text-generation
---

# nemotron-labs-dory1.0-4b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | dory |
| **Source** | https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B |
| **License** | other (openmdw-1.1) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 `export_hf` with a model definition of its own (transformers has no Dory class; the formulas follow NVIDIA's vLLM pull request); repro = hf-to-litertlm dory_work/convert_dory_4b.sh (REPRODUCE.md "NVIDIA-Nemotron-Labs-Dory1.0-4B") (card L24, L130) litert-torch 0.9.4, ai-edge-quantizer 0.9.0, litert-converter 0.4.0, torch 2.13.0, transformers 5.14.1; packed with litert-lm-builder 0.18.0 (card L130) |
| **Command** | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 rev b366b87f -> export with the 20 recurrent pairs unrolled twice (weights stored once), per-loop KV caches (56), prefill signatures 128 / 512 / 1024 plus decode over a 2048-token KV budget, the embedding table in its own section; the MLP normalization of 11 of the 36 MLP blocks scaled by 2^-k (k = 1-3) before the sum with the reciprocal square root capped (the same numbers in fp32) -> int8 dynamic (channelwise int8 weights, float activations) on the linear layers, int8 embedding -> metadata (NVIDIA's template with three runtime edits, no start token, stop tokens </s> and <|im_end|> plus 14 stop sequences, the thought channel, max_num_tokens 2048) (card L33, L130-L137)` |
| **Quantization** | int8 dynamic (channelwise int8 weights, float activations) on the linear layers; int8 embedding in its own section (card L33) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `NVIDIA-Nemotron-Labs-Dory1.0-4B_int8.litertlm` | `25632f91399cf7408d546c1c0531290d1c9ca1b7d4e9b75826600bfc2f9f6083` | 3859.21 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-10-22/nemotron-labs-dory1.0-4b-int8__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-22/nemotron-labs-dory1.0-4b-int8__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-22/nemotron-labs-dory1.0-4b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-22/nemotron-labs-dory1.0-4b-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | sig:Android library (hfmodels-android samples/chat ChatDeviceCheck, debug build); one conversation, 'What is 17 + 25? Answer briefly.' (expect '42') | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.1 · Android 16 (BP4A.251205.006.S942QOPS1AZH9) | 2026-10-22 | measured |
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 83.025 | 5.0625 | 12600.0 | 5852.553 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-22 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 302.035 | 15.7 | 3528.2 | 22533.094 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-22 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 12.54 | 1.18 | 83930.0 | 4932.665 | Pixel 8a · Google Tensor G3 · litert-lm 0.18.0 · Android 16 (CP1A.260505.005) | 2026-10-22 | measured |

## Pitfalls

- CPU file: use the int8 file on a CPU and the int4 file on a GPU. An earlier int4 build prefilled at a third of the int8 rate on the Galaxy S26 CPU (29.0 against 92.1 tok/s), and the int4 weights' greedy answers change from one CPU to another (card L36, L50).
- Accuracy: GSM8K questions 0-99, thinking off, greedy, at most 2,048 tokens of prompt and reply: this file 93 / 100 on the Mac CPU, the BF16 reference (llama.cpp, two loops) 95 and 86 with one loop; with thinking on the 2048-token budget is short for GSM8K (the two-loop reference left 29 of the first 32 questions with the thought still open) (card L36, L58-L68).
- Phones: the CPU was capped through almost every sample of every CPU run (sustained rates); the int8 file on a GPU was not run on the phones: on the 12 GB Galaxy S26 an earlier int4 build with fp32 activations already left 1.47 GB of MemAvailable and the int8 weights are 1.6 GB larger (card L122, L126).
- NPU: a static-range build (int16 activations, int8 weights) on the Galaxy S26: the Qualcomm plugin took the whole graph, but within 9 s of the on-phone compile free memory fell from 6.93 to 0.92 GB and the harness stopped it; an ahead-of-time build was not tried; there is no NPU file (card L125).
- Thinking and template: thinking is on by default (the generation prompt ends with <think> and a newline, the bundle declares the thought channel); rendered with enable_thinking false the prompt ends with <think></think>; NVIDIA recommends temperature 1.0 and top-p 0.95, the bundle sets no sampler; model type generic_model (tool calls not parsed; tool calling not tested) (card L97, L99).
- Final mark on LiteRT-LM 0.16.1 and earlier: of the 14 stop sequences, 13 put one of 13 characters (12 punctuation marks and a space) in front of <|im_end|> and a newline; an answer whose last mark is a token of its own, such as a ? or . after a word, arrives without that mark on LiteRT-LM 0.16.1 (the runtime of the Android app under Loops) and on 0.11.0 to 0.16.0 (source read): the runtime holds the mark back, the answer ends on <|im_end|> alone and the held mark is never written; a mark the tokenizer merges with the text before it (**., ).) is kept; 0.17.0 and later write the held mark (card L97).
- Second turn with thinking off: from the second turn of a conversation on, LiteRT-LM puts the previous answer's stop token into the KV cache with a zero embedding, because the file holds the embedding table in its own section and the decoder takes embeddings (read in the source of 0.16.0 to 0.18.1; measured with 0.18.1 on WeVisOne-VL-2B and Qwen2.5-Coder-1.5B-Instruct, not on these files), so later answers can differ from the answer to the same messages in one prompt; with thinking on the runtime rebuilds the cache at each turn, and the source shows a zero embedding at another token from the third turn on (not measured); one question per conversation, or a new conversation that starts with the earlier messages and the new question, avoids both (card L101).
- Conversation preface with tools and no messages: the template reads messages[0] without checking that the list is non-empty, so LiteRT-LM's render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages; read in the source of 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on these files' template text; the default path, where the first message is rendered with the preface, is not affected; a system message or any initial message in the conversation config avoids it (card L141).
- Cache: without --cache no, the CLI can write a compiled cache file next to the model on its first run; without --backend, `run` uses the CPU (card L95).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
