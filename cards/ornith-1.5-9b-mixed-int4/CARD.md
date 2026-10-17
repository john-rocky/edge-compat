---
family: ornith
license: mit
model_id: ornith-1.5-9b-mixed-int4
source_url: https://huggingface.co/litert-community/Ornith-1.5-9B
task: text-generation
---

# ornith-1.5-9b-mixed-int4

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
| **Quantization** | Mixed INT4: int4 blockwise-32 min-max on the linear layers, int8 on the lm_head and the embedding, the 48 decay/gate projections float; 216,006,656 block scales, all normal floats (none zero or subnormal); fp32 GPU activations declared (card L32, L107) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ornith-1.5-9B_mixed_int4.litertlm` | `e4cad4a90869a2f6589a0f47268016e5ce5ac438ff8b22d68ff9b82b2e0183b4` | 5779.607 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-15/ornith-1.5-9b-mixed-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ornith-1.5-9b-mixed-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 68.055 | 14.78 | 15122.4 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 615.54 | 45.58 | 1685.6 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Text only: the decoder of the multimodal release; no vision tower, no MTP head. Only the 8 attention layers keep a growing KV cache (a 4096-token budget); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L25).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L21).
- Thinking off, no switch: the template writes the empty think block that Ornith's template writes at enable_thinking=False into every assistant turn and the generation prompt; a switch template would drop the think blocks Ornith writes into past turns, so none is shipped (card L27, L77).
- One stop token, <|im_end|> (248046): the repo has no generation config and its model config lists that one end token (card L77).
- Tokenizer: the tokenizer section is the repo's tokenizer.json byte for byte (174 of 174 probes equal the tokenizers library on litert-lm 0.18.0 and 0.17.1) (card L105).
- Galaxy S26 GPU: the engine of this file does not build on a phone of this memory class: on a device-lab Galaxy S26 its benchmark session ended during the engine build after the delegate had taken prefill_1024 / prefill_256 / prefill_64, with an allocator failure and the low-memory killer removing other apps, as the same-recipe Ornith-1.0-9B files did in four sessions; the CPU was not run (12.1 GB by the earlier CPU peak factor against 11.66 GB) (card L38, L98).
- fp32 GPU activations are declared (the int8 sibling answered 0 of 8 with the runtime's fp16 default) (card L47).
- Tool calling: the model writes the call in Qwen3.5's XML form as text only when given Ornith's own tool block; the runtime does not parse or run it (LiteRT-LM issue 3927); a tool result sent back as a `tool` message was not used (card L79).
- One Engine per conversation (LiteRT-LM issue 3165): no carry-over seen in the probe on this file, but the recurrent state is carried all the same (card L81).
- Long outputs: 3 of 8 replies of a 512-token long-output check reached the cap (answers longer than 512 tokens, cut mid-sentence, none repeating); no reply looped (card L51).
- Pixel 8a: not run (no Qwen3.5 Hub file built a GPU engine on this Mali phone under a memory guard). iOS not measured: one 5.64 GiB TFLite section, over the ~2 GiB single-section limit (card L43, L99).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, Mac GPU): bf16 94 / Mixed INT4 92, agreeing on 92 of 100 questions; the file is 34 % smaller than the int8 file (card L34, L57-L63).
- Mac: the faster file on the GPU (7.28 vs 8.16 s, decode 45.6 vs 39.6 tok/s) and the slower on the CPU (blockwise prefill 68 tok/s against 234); without --cache, the first CPU `run` writes a 4.96 GB compiled-weights cache next to the model; a three-signature variant passed the Mac 8 questions but did not build on the Galaxy S26 either (card L38-L43, L75, L94).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
