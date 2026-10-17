---
family: ornith
license: mit
model_id: ornith-1.0-9b-int8
source_url: https://huggingface.co/litert-community/Ornith-1.0-9B
task: text-generation
---

# ornith-1.0-9b-int8

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
| **Quantization** | dynamic int8 on every FULLY_CONNECTED + int8 embedding; the convs, the delta rule and the 48 GatedDeltaNet decay/gate projections (in_proj_a / in_proj_b) float; fp32 GPU activations declared (card L31, L106) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ornith-1.0-9B_int8.litertlm` | `b42d5c1e7393b26b2006e79d2f339b2051c6bae0a7e49f48981f7bcbe37e122f` | 8774.879 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-15/ornith-1.0-9b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ornith-1.0-9b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 227.365 | 15.17 | 4586.3 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 581.385 | 39.99 | 1788.8 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Text only: the decoder of the multimodal release (8,953,803,264 of its 9,409,813,744 parameters); no image input. Only the 8 attention layers keep a growing KV cache (a 4096-token budget; Ornith recommends 262,144); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L23, L25).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L21).
- Thinking off: Ornith-1.0-9B is a reasoning model, and the bundle's 701-byte template writes the empty think block into every assistant turn and the generation prompt (Ornith's enable_thinking=False form) with no thinking switch; the Qwen3.5-4B conversion's switch template drops the think block that Ornith's template writes into past assistant turns, so its multi-turn prompts would differ from the form Ornith trained on (card L27, L77).
- Template: byte-identical to Ornith's enable_thinking=False render in six conversation shapes without tools; the two tool shapes differ (no tool list, a tool message becomes a plain `tool` turn); no start token; stop ids 248044 and 248046 from Ornith's generation config; the tokenizer section is the repo's tokenizer.json byte for byte (card L49, L77).
- fp32 GPU activations are declared on purpose: with the runtime's fp16 default the int8 weights answered 0 of 8 check questions (every reply a run of `!`) and printed WebGPU validation errors, for 2.8 % less time (card L47, L63).
- Tool calling: a tool passed through the API changes nothing in the prompt (the template renders no tool list); with Ornith's own `# Tools` block as the system message the model writes the call in Qwen3.5's XML form as text, which the runtime (generic model type) does not parse or run (LiteRT-LM issue 3927); an app parses that text itself (card L79).
- One Engine per conversation: litert-lm 0.18.0 starts a new Conversation on a used Engine from the earlier conversation's recurrent state (LiteRT-LM issue 3165); the probe on these files gave the same answers on a shared and a fresh Engine, but the state is there all the same (card L81).
- Init: litert-lm 0.18.0's CLI prints an init time that sums overlapping init phases, about twice the engine build in the log's timestamps (card L85).
- Galaxy S26: the int8 file was not started on the phone's GPU after a sibling 8B-class int8 file restarted this phone during its GPU engine warm-up (2026-10-14); the CPU was not run (by earlier CPU peak factors the file would need 13.8 GB against the phone's 11.66 GB) (card L98).
- Pixel 8a: not run; no Qwen3.5 Hub file built a GPU engine on this 7.75 GB Mali phone under a memory guard (card L99). iOS not measured: the file is one 8.57 GiB TFLite section, over the ~2 GiB single-section limit (card L43).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, a fresh process per question, Mac GPU): bf16 95 / int8 97; the int8 file agrees with bf16 on 98 of 100 questions (card L34, L55-L63).
- Mac: the int8 file is the faster file on the CPU (prefill 227 vs 68 tok/s for the Mixed INT4 file), the Mixed INT4 file on the GPU; without --cache, `run` writes a 7.99 GB compiled-weights cache next to the model on its first CPU run; 50.5 GB of memory during the CPU benchmark (card L43, L75, L94).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
