---
family: qwen3
license: apache-2.0
model_id: oc-e-non-v1-mixed-int4b32-fp16act-ekv4096
source_url: https://huggingface.co/mlboydaisuke/Oc-e-Non-v1-LiteRT
task: text-generation
---

# oc-e-non-v1-mixed-int4b32-fp16act-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/mlboydaisuke/Oc-e-Non-v1-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 export_hf with its GPU composites, quantized with ai-edge-quantizer 0.10.1; one command rebuilds both files (the recipe in john-rocky/hf-to-litertlm REPRODUCE.md) (card L18, L69, L71) litert-torch 0.10.0, ai-edge-quantizer 0.10.1; litert_lm_advanced_main built from LiteRT-LM v0.19.0 (771b06e1) for the phone numbers, the PyPI litert-lm 0.19.2 on an Apple M4 Max for the accuracy checks (card L18, L42, L52, L65) |
| **Command** | `p-e-w/Oc-e-Non-v1 rev dbf4e269 (bf16) -> export_hf with its GPU composites; int4 blockwise-32 weights for the linear layers, int8 token table and output layer; fp16 GPU activations; LiteRT-LM 0.19.1's Qwen3 configuration with the checkpoint's sampler (top-k 20, top-p 0.8, temperature 0.7), <|endoftext|> as a second stop token and no thought channel; the checkpoint's tokenizer.json byte for byte (card L35, L69-L70)` |
| **Quantization** | int4 (blocks of 32) for the linear layers; int8 token table and output layer; fp16 GPU activations (card L35) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `oc_e_non_v1_mixed_int4b32_fp16act_ekv4096.litertlm` | `0dac09e4d6d298216444caf4a0ad0b9a29e71d31953d9568cd0e58de01d7a854` | 2910.547 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.0/2026-11-15/oc-e-non-v1-mixed-int4b32-fp16act-ekv4096__galaxy-s26.json`, `data/device_runs/0.19.0/2026-11-15/oc-e-non-v1-mixed-int4b32-fp16act-ekv4096__pixel-11-pro.json`, `data/device_runs/0.19.2/2026-11-15/oc-e-non-v1-mixed-int4b32-fp16act-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 36.065 | 9.125 | 28505.0 | 6225.883 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-15 | measured |
| galaxy-s26 | gpu | pass | yes | pass | sig:the first 8 of the 12 check questions of line 3 (17 + 25, the capital of Japan, the opposite of 'hot', the days of a week, 'thank you' in French, 8 times 7, 0.9 or 0.11, the rhyme), each with ' Answer briefly.' in its own process, --max_num_tokens=1024 --max_output_tokens=256, no sampler flags (the file's sampler); each verdict by the question's regex against Transformers (float32, greedy) | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-15 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 222.1125 | 18.105 | 4707.5 | 1727.824 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.19.0 · Android 16 (BP4A.251205.006) | 2026-11-15 | measured |
| mac-studio-m4-max | cpu | pass | - | pass | sig:the 12 check questions of line 3 (17 + 25, the capital of Japan, the opposite of 'hot', the days of a week, 'thank you' in French, 8 times 7, 0.9 or 0.11, the rhyme, a word problem, fib(10) in Python, the planets in order, a Japanese sentence in English), each with ' Answer briefly.' in a new conversation, and a 3-turn conversation, greedy, at most 1,024 new tokens per reply; verdicts against Transformers (float32, greedy) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.2 · macOS 27.0 (26A428) | 2026-11-15 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:GSM8K test, the first 100 questions with a chain-of-thought prompt, one conversation each, greedy, at most 512 new tokens; paired against Transformers (float32, greedy): b = right in the reference and wrong in the file, c = the reverse, pass = b - c <= 2 sqrt(b + c) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.2 · macOS 27.0 (26A428) | 2026-11-15 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the 12 check questions of line 3 (17 + 25, the capital of Japan, the opposite of 'hot', the days of a week, 'thank you' in French, 8 times 7, 0.9 or 0.11, the rhyme, a word problem, fib(10) in Python, the planets in order, a Japanese sentence in English), each with ' Answer briefly.' in a new conversation, and a 3-turn conversation, greedy, at most 1,024 new tokens per reply; verdicts against Transformers (float32, greedy) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.2 · macOS 27.0 (26A428) | 2026-11-15 | measured |
| pixel-11-pro | cpu | fallback | no | pass | sig:the first 8 of the 12 check questions of line 3 (17 + 25, the capital of Japan, the opposite of 'hot', the days of a week, 'thank you' in French, 8 times 7, 0.9 or 0.11, the rhyme), each with ' Answer briefly.' in its own process, --max_num_tokens=1024 --max_output_tokens=256, no sampler flags (the file's sampler); each verdict by the question's regex against Transformers (float32, greedy) | - | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-15 | measured |
| pixel-11-pro | gpu | pass | yes | pass | sig:the first 8 of the 12 check questions of line 3 (17 + 25, the capital of Japan, the opposite of 'hot', the days of a week, 'thank you' in French, 8 times 7, 0.9 or 0.11, the rhyme), each with ' Answer briefly.' in its own process, --max_num_tokens=1024 --max_output_tokens=256, no sampler flags (the file's sampler); each verdict by the question's regex against Transformers (float32, greedy) | - | - | - | - | - | Pixel 11 Pro · Google Tensor G6 · litert-lm 0.19.0 · Android 17 (CD1A.260905.001.B1) | 2026-11-15 | measured |

## Pitfalls

- Use LiteRT-LM 0.19.0 or newer: the files were checked on 0.19.0 (Android) and 0.19.2 (Mac); earlier releases were not tried, nor phones with 8 GB of memory (card L25).
- An incomplete download: if `run` stops right after the download with no error message from the CLI (a bus error; exit status 138 on a Mac) or with an error log that contains `too large for file size`, the litert-lm CLI (tested on 0.19.2 and litert-lm-nightly 0.20.0.dev20261112; 0.14.0 to 0.19.1 have the same download code) can keep an incomplete download in ~/.litert-lm/cache/huggingface/mlboydaisuke/Oc-e-Non-v1-LiteRT/ and use it again without checking its size; delete that file and run the command again (card L27).
- Question 10 of the Mac checks: the file writes a correct Python function and then gives fib(10) as 10, where the original model gives 55, on both the Mac GPU and CPU (11 of 12 verdicts equal) (card L47-L48, L52).
- Two processes of this file on the Mac GPU gave 75 and 83 on GSM8K (the first 100), with the same text in 39 of the 100 replies: on LiteRT-LM's default Mac GPU path, where the GPU sampler writes the next decode step's input (sampler_handles_input), greedy replies to long prompts have also differed from run to run with other models (card L47, L52).
- 4,096 tokens of context: the prompt, the reply and earlier turns together (card L38); at --max_num_tokens=4096, the file's own context, two Galaxy S26 GPU runs decoded 17.91 and 18.26 tokens/s, the card's row is at 2048 (card L65).
- The Galaxy S26 peak (VmHWM) does not count the memory the GPU driver holds (card L65).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
