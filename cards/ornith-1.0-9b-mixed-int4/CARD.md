---
family: ornith
license: mit
model_id: ornith-1.0-9b-mixed-int4
source_url: https://huggingface.co/litert-community/Ornith-1.0-9B
task: text-generation
---

# ornith-1.0-9b-mixed-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | ornith |
| **Source** | https://huggingface.co/litert-community/Ornith-1.0-9B |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (pinned commit 115a1360 + the Qwen3.5 hybrid patch of hf-to-litertlm, 2026-08-21 version; lightweight export, constants of 1e8 bytes and more stay lazy) + post-hoc ai-edge-quantizer; repro = hf-to-litertlm qwen35_work/ litert-torch 115a1360 + patch, transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0; packed and checked with litert-lm 0.18.0 |
| **Command** | `qwen35_work/convert_qwen35_9b.sh (hf-to-litertlm; ornith-ai/Ornith-1.0-9B at 83dc1f5e = the Qwen3.5-9B recipe run with this repo's id and revision): text-decoder export (6 prefill signatures 1024..1 + decode, a 4096-position KV budget, 64 state buffers) -> int8 / Mixed INT4 quantization with the 48 in_proj_a / in_proj_b projections float -> fp32 GPU activations declared -> the thinking-off template (the only metadata field changed); run as published on 2026-10-16 it rebuilt all four sections of both files byte for byte` |
| **Quantization** | Mixed INT4: int4 blockwise-32 min-max on the linear layers, int8 on the lm_head and the embedding, the 48 decay/gate projections float; 216,006,656 block scales, all normal floats (none zero or subnormal); fp32 GPU activations declared (card L32, L106) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ornith-1.0-9B_mixed_int4.litertlm` | `bf01eff637705a82c1a7dab65c922b9bab5a8fdb445ffeb6ac6aa67c860ee80c` | 5779.889 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-15/ornith-1.0-9b-mixed-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ornith-1.0-9b-mixed-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 68.44 | 15.765 | 15049.3 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 585.355 | 44.925 | 1773.5 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Text only: the decoder of the multimodal release; no image input. Only the 8 attention layers keep a growing KV cache (a 4096-token budget; Ornith recommends 262,144); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L23, L25).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L21).
- Thinking off: the bundle's 701-byte template writes the empty think block into every assistant turn and the generation prompt (Ornith's enable_thinking=False form) with no thinking switch, because the Qwen3.5-4B conversion's switch template drops the think block that Ornith's template writes into past assistant turns (card L27, L77).
- Template: byte-identical to Ornith's enable_thinking=False render in six conversation shapes without tools; the two tool shapes differ; no start token; stop ids 248044 and 248046; the tokenizer section is the repo's tokenizer.json byte for byte (card L49, L77).
- fp32 GPU activations are declared on both files: with the runtime's fp16 default the int8 weights answered 0 of 8 check questions (every reply a run of `!`) (card L47, L63).
- Tool calling: a tool passed through the API changes nothing in the prompt; with Ornith's own `# Tools` block as the system message the model writes the call in Qwen3.5's XML form as text, which the runtime (generic model type) does not parse or run (LiteRT-LM issue 3927) (card L79).
- One Engine per conversation: litert-lm 0.18.0 starts a new Conversation on a used Engine from the earlier conversation's recurrent state (LiteRT-LM issue 3165); the probe on these files showed no carry-over, unlike the Qwen3.5-4B conversion's Mixed INT4 file (card L81).
- Galaxy S26: the GPU engine of this file does not build on a phone of the S26's memory class: on a platform device-lab Galaxy S26 (SM-S942U1, 2026-10-16) four sessions of this file and a three-signature variant ended during the engine build while the low-memory killer ran, with no output; the own phone was not tried; the CPU was not run (12.1 GB by the CPU peak factor against the phone's 11.66 GB) (card L94, L98).
- Pixel 8a: not run; no Qwen3.5 Hub file built a GPU engine on this 7.75 GB Mali phone under a memory guard (card L99). iOS not measured: one 5.64 GiB TFLite section, over the ~2 GiB single-section limit (card L43).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, a fresh process per question, Mac GPU): bf16 95 / Mixed INT4 95; the file agrees with bf16 on 92 of 100 questions (right only in bf16 4, right only in this file 4) (card L34, L55-L63).
- Mac: the faster file on the GPU (8.8 % less time than int8, decode 44.9 vs 40.0 tok/s at about the same prefill rate), the slower one on the CPU (its blockwise prefill runs at 68 tok/s against 227); without --cache, `run` writes a 4.96 GB compiled-weights cache next to the model; 30.1 GB of memory during the CPU benchmark (card L43, L75, L94).
- Three-turn check: three of four runs gave the bf16 model's replies word for word; this file on the GPU chose another word ("Enormous") and kept it across the turns (card L49).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
