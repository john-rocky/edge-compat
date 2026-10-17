---
family: ornith
license: mit
model_id: ornith-1.5-9b-int8
source_url: https://huggingface.co/litert-community/Ornith-1.5-9B
task: text-generation
---

# ornith-1.5-9b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | ornith |
| **Source** | https://huggingface.co/litert-community/Ornith-1.5-9B |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (pinned commit 115a1360 + the Qwen3.5 hybrid patch of hf-to-litertlm, 2026-08-21 version; lightweight export with a 1e8-byte lazy-constant threshold) + post-hoc ai-edge-quantizer; repro = hf-to-litertlm qwen35_work/ litert-torch 115a1360 + patch, transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0; packed and checked with litert-lm 0.18.0 |
| **Command** | `qwen35_work/convert_qwen35_9b.sh (hf-to-litertlm; ornith-ai/Ornith-1.5-9B at 489cb979, the same script bytes as the Qwen3.5-9B and Ornith-1.0-9B runs): text-decoder export (6 prefill signatures 1024..1 + decode, a 4096-position KV budget, 64 state buffers) -> int8 / Mixed INT4 quantization with the 48 in_proj_a / in_proj_b projections float -> fp32 GPU activations declared -> the thinking-off template; then the tokenizer step (tools/tokenizer_parity/swap_tokenizer_section.py: the repo's tokenizer.json as the tokenizer section, the other three sections byte-identical)` |
| **Quantization** | dynamic int8 on every FULLY_CONNECTED + int8 embedding; the convs, the delta rule and the 48 GatedDeltaNet decay/gate projections (in_proj_a / in_proj_b) float; fp32 GPU activations declared (card L31, L107) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ornith-1.5-9B_int8.litertlm` | `f21058268a2b66a351007a1c3268dba79387943ea27cb595e6203c3b36d05586` | 8774.598 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-15/ornith-1.5-9b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ornith-1.5-9b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 234.135 | 14.975 | 4442.8 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 608.97 | 39.565 | 1706.9 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Text only: the decoder of the multimodal release (8,953,803,264 of its 9,653,104,368 parameters; no vision tower, no MTP head). Only the 8 attention layers keep a growing KV cache (a 4096-token budget); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L25).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L21).
- Thinking off, no switch: the template writes the empty think block that Ornith's template writes at enable_thinking=False into every assistant turn and the generation prompt; a switch template would drop the think blocks Ornith writes into past turns, so none is shipped (card L27, L77).
- One stop token, <|im_end|> (248046): the repo has no generation config and its model config lists that one end token (the Qwen3.5-9B and Ornith-1.0-9B conversions also stop on <|endoftext|>, 248044) (card L77).
- Tokenizer: the export writes transformers' Qwen2Tokenizer pipeline (no \p{M} in the pre-tokenizer) while the repo's tokenizer.json has one; the section was replaced with the repo file, and the runtime's ids equal the tokenizers library's on 174 of 174 probes on litert-lm 0.18.0 and 0.17.1 (card L105).
- CPU prompt-length check 15 of 17: at 23 and 24 tokens this file's CPU answered with a polite refusal instead of the asked word, the same in three fresh processes each, while its GPU and the bf16 model kept the word (the bf16 refusal a close second, margins 1.6 / 1.1 logits); the cause is not measured (card L49).
- fp32 GPU activations are declared: with the runtime's fp16 default this file answered 0 of 8 check questions (every reply a run of `!`) for 4.5 % less time (card L47, L63).
- Tool calling: a tool passed through the API is not rendered (no tool list in the template); with Ornith's own tool block the model writes the call in Qwen3.5's XML form as text, which the runtime does not parse or run (LiteRT-LM issue 3927); a tool result sent back as a `tool` message was not used (card L79).
- One Engine per conversation: the runtime starts a new Conversation from the earlier one's recurrent state (LiteRT-LM issue 3165); the probe on these files gave the same answers on a shared and a fresh Engine, but the state is carried all the same (card L81).
- Galaxy S26: the int8 file was not started on the phone's GPU (a sibling 8B-class int8 file restarted the phone during its GPU warm-up) and the CPU was not run (13.8 GB by the earlier CPU peak factor against 11.66 GB); Pixel 8a not run; not run on an NPU (card L98-L100).
- iOS not measured: one 8.57 GiB TFLite section, over the ~2 GiB single-section limit (card L43).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, Mac GPU): bf16 94 / int8 95, agreeing on 93 of 100 questions (card L57-L63).
- Mac: the faster file on the CPU (21.47 vs 32.37 s; prefill 234 vs 68 tok/s) and the slower on the GPU (8.16 vs 7.28 s); without --cache, the first CPU `run` writes a 7.99 GB compiled-weights cache next to the model; the CLI's printed init sums overlapping init phases (card L38-L43, L75, L85).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
