---
family: lfm2
license: other (lfm1.0)
model_id: lfm2-2.6b-exp-int8
source_url: https://huggingface.co/litert-community/LFM2-2.6B-Exp
task: text-generation
---

# lfm2-2.6b-exp-int8

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
| **Command** | `LiquidAI/LFM2-2.6B-Exp rev 2251c865: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache) -> post-hoc int8 quantization -> metadata repack with LiteRT-LM's lfm2 configuration (card L111-L112)` |
| **Quantization** | int8 dynamic on linears + int8 embedding, convs float (card L27) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2-2.6B-Exp_int8.litertlm` | `a44c3cfd0644e9fe9036c0efe3955df48a91990fe6e42855a9de0020c4e76258` | 2597.627 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-13/lfm2-2.6b-exp-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-12/lfm2-2.6b-exp-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-exp-int8__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-exp-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 285.63 | 18.545 | 3645.0 | 4163.8 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 1140.525 | 20.435 | 950.0 | 779.434 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | run_failed | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 377.26 | 38.27 | 2740.5 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 2441.36 | 123.39 | 427.5 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 70.19 | 5.845 | 15045.0 | 3659.46 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 498.545 | 11.095 | 2145.0 | 1667.16 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; on litert-lm 0.17.1 this file answered "What is the capital of France?" as if no question had been asked (one CPU run) (card L21).
- Reasoning trace: Liquid AI's card says nothing about reasoning, yet the model sometimes writes a trace between <think> and </think>, which goes to the thought channel (printed by the CLI between [thought] and [/thought]); one appeared for "What is the capital of France?" (this file, CPU and GPU; with a 16-token output limit the reply was empty, all 16 tokens being trace) and for 1 of 120 IFEval-lite prompts; --thinking-budget 0 and 32 did not suppress it (card L23, L72).
- Template: Liquid AI's chat template (the LFM2-2.6B file) fails at render on litert-lm 0.18.0 (map has no method named get), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|> (card L72, L113).
- Tool calling is declared (model type lfm2): the file returned a parsed get_weather call (Python API, CPU, greedy) and used its result with automatic tool calling; with tools the template omits Liquid AI's tool-list and tool-response markers, wraps the tool result in [ … ] and quotes call arguments with double quotes (card L74).
- One Engine per conversation: with the same system prompt in two conversations and user messages beginning with the same eight words (the condition of LiteRT-LM issue #3165), a shared Engine answered "The third item in the list is cherry." where a fresh Engine answered "cherry" (3 of 3 each, both correct); with different system prompts the answers matched a fresh Engine (card L76).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0; Liquid AI recommends temperature 0.3, min_p 0.15 and repetition_penalty 1.05, which the bundle does not carry; --top-k 1 gives greedy output (card L70).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (2.58 GB for this file); without --backend, `run` uses the CPU; on the Mac CPU this file ended normally on the dates prompt in 3 of 3 runs (card L68).
- Speed: int4 is the faster file on every GPU and this file on every CPU (Mac 9.40 vs 11.11 s, Galaxy S26 17.39 vs 25.91 s, Pixel 8a 58.42 vs 65.57 s for a 1024-token prefill plus 256 decoded tokens); not run on the NPU (card L32-L39, L107).
- iOS not measured: this file's one weights section is 2,722,843,088 bytes (2.54 GiB); the LFM2.5-2.6B card reports an int8 file with a 2.87 GB weights section running on an iPhone 17 Pro only with the extended-virtual-addressing and increased-memory entitlements (card L39).
- Phone conditions: on the Galaxy S26 the GPU clock ceiling fell from 1300 to 826 MHz inside this file's GPU benchmark and the CPU clock was lowered in 23 of 24 samples of its CPU run; on the Pixel 8a the CPU clock was lowered in 51 of 53 samples of the CPU run, and in the GPU benchmark the lowest MemAvailable sample was 501,160 kB (card L98, L107).
- Accuracy (greedy, at most 2048 output tokens, litert-lm on the Mac GPU, comparable only within the card's table; Liquid AI publishes no benchmark table for this checkpoint): GSM8K 87 of 100 (bf16 89), IFEval-lite 102 of 120 (bf16 102); 8-question check 7/8 on CPU and GPU, bf16 7/8, the same rhyme question missed (card L43, L49-L57).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L117).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
