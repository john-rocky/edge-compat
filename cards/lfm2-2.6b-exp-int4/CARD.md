---
family: lfm2
license: other (lfm1.0)
model_id: lfm2-2.6b-exp-int4
source_url: https://huggingface.co/litert-community/LFM2-2.6B-Exp
task: text-generation
---

# lfm2-2.6b-exp-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2 |
| **Source** | https://huggingface.co/litert-community/LFM2-2.6B-Exp |
| **License** | other (lfm1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4; repro = hf-to-litertlm lfm_work/convert_lfm2_26b_exp.sh (card L111) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; packed by the litert-lm 0.18.0 CLI |
| **Command** | `LiquidAI/LFM2-2.6B-Exp rev 2251c865: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache) -> post-hoc int4 quantization -> int4 zero-scale fix (746,432 zero block scales in 26 tensors, replaced by each tensor's smallest nonzero scale, block payload zeroed) -> metadata repack with LiteRT-LM's lfm2 configuration (card L111-L112)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 lm_head + embedding (card L28) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2-2.6B-Exp_int4.litertlm` | `1d3401ca1b81dc40de154b88a13b654d8ac1838281fd47303aa161f0b825f877` | 1458.722 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-12/lfm2-2.6b-exp-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-exp-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-exp-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 74.035 | 21.215 | 13995.0 | 3111.75 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 664.285 | 27.615 | 1585.0 | 1029.22 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 177.07 | 48.04 | 5808.5 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 2475.63 | 161.77 | 419.8 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 30.525 | 8.005 | 33715.0 | 2597.15 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 171.73 | 15.9 | 6030.0 | 1504.58 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |

## Pitfalls

- Mac CPU repetition: on the Mac CPU this file is not reliable with the card's dates prompt - in 9 runs, 3 repeated lines until the 4096-token context was full (2 of 4 with the bundle's sampler, 1 of 5 with --top-k 1) and 1 kept correcting itself; on the GPU it ended normally in 4 of 4 runs; the cause was not established (card L39, L68).
- Runtime: tested on litert-lm 0.18.0 on a Mac; this file was not tried on litert-lm 0.17.1, where the int8 file of the same conversion answered as if no question had been asked (card L21).
- Block scales: the plain int4 quantization left 746,432 zero block scales in 26 tensors; this file replaces them by each tensor's smallest nonzero scale with the block payload zeroed (card L112).
- Reasoning trace: Liquid AI's card says nothing about reasoning, yet the model sometimes writes a trace between <think> and </think>, which goes to the thought channel; in the card's checks it appeared for 1 of 120 IFEval-lite prompts per row and none for GSM8K, the checks or the phone prompts (card L23, L72).
- Template: Liquid AI's chat template (the LFM2-2.6B file) fails at render on litert-lm 0.18.0 (map has no method named get), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|> (card L72, L113).
- Tool calling is declared (model type lfm2): the file returned a parsed get_weather call (Python API, CPU, greedy) and used its result with automatic tool calling; with tools the template omits Liquid AI's tool-list and tool-response markers, wraps the tool result in [ … ] and quotes call arguments with double quotes (card L74).
- One Engine per conversation: on the int8 file, a shared Engine changed an answer's wording under the condition of LiteRT-LM issue #3165 (the same system prompt, user messages beginning with the same eight words); the card advises a new Engine for each conversation (card L76).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0; Liquid AI recommends temperature 0.3, min_p 0.15 and repetition_penalty 1.05, which the bundle does not carry; --top-k 1 gives greedy output (card L70).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (1.51 GB for this file); without --backend, `run` uses the CPU (card L68).
- Speed: this file is the faster one on every GPU (Mac 2.00 vs 2.49 s, Galaxy S26 10.81 vs 13.42 s, Pixel 8a 22.06 vs 25.14 s for a 1024-token prefill plus 256 decoded tokens) and the slower on every CPU; 43.8 % smaller than the int8 file; not measured on iOS, not run on the NPU (card L30-L39, L107).
- Phone conditions: on the Galaxy S26 the GPU clock ceiling fell from 1300 to 726 MHz inside this file's GPU benchmark and the CPU clock was lowered in 36 of 37 samples of its CPU run; on the Pixel 8a the CPU clock was lowered in 65 of 65 samples of the CPU run (card L98, L107).
- Accuracy (greedy, at most 2048 output tokens, litert-lm on the Mac GPU, comparable only within the card's table; Liquid AI publishes no benchmark table for this checkpoint): GSM8K 89 of 100 (bf16 89), IFEval-lite 101 of 120 (bf16 102); 8-question check 7/8 on CPU and GPU, bf16 7/8, the same rhyme question missed (card L43, L49-L57).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L117).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
