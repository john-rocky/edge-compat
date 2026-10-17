---
family: qwen3.5
license: apache-2.0
model_id: qwen35-9b
source_url: https://huggingface.co/litert-community/Qwen3.5-9B
task: text-generation
---

# qwen35-9b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3.5 |
| **Source** | https://huggingface.co/litert-community/Qwen3.5-9B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (pinned commit 115a1360 + the Qwen3.5 hybrid patch of hf-to-litertlm, 2026-08-21 version; lightweight export with a 1e8-byte lazy-constant threshold) + post-hoc ai-edge-quantizer; repro = hf-to-litertlm qwen35_work/ litert-torch 115a1360 + patch, transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0; packed and checked with litert-lm 0.18.0 |
| **Command** | `qwen35_work/convert_qwen35_9b.sh (hf-to-litertlm; Qwen/Qwen3.5-9B at c2022362): text-decoder export (6 prefill signatures 1024..1 + decode, a 4096-position KV budget, 64 state buffers) -> int8 / Mixed INT4 quantization with the 48 in_proj_a / in_proj_b projections float -> fp32 GPU activations declared; then the tokenizer step (the repo's tokenizer.json as the tokenizer section) and qwen35_work/use_switch_template.sh (the Qwen3.5-4B files' template, thinking switch and thought channel)` |
| **Quantization** | dynamic int8 on every FULLY_CONNECTED + int8 embedding; the convs, the delta rule and the 48 GatedDeltaNet decay/gate projections (in_proj_a / in_proj_b) float; fp32 GPU activations declared (card L30, L108) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-9B_int8.litertlm` | `82d23246c88d957ab316960147be5e04adc7b3f4492548406dcd6540fbeb92f1` | 8774.598 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-16/qwen35-9b__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-16/qwen35-9b__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | sig:8-question gate (CLI run --cache disk) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:one prompt (CLI run --cache no) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:three-turn conversation, thinking off (Python API) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:three-turn conversation, thinking on with thinking_token_budget 300 (Python API) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | sig:8-question gate (CLI run --backend gpu --cache no) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:three-turn conversation, thinking off (Python API) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:three-turn conversation, thinking on with thinking_token_budget 300 (Python API) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |

## Pitfalls

- Text only: the decoder of the multimodal release (8,953,803,264 of its 9,653,104,368 parameters); no image or video input. Only the 8 attention layers keep a growing KV cache (a 4096-token budget); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L24).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L20).
- Thinking is off unless asked: the bundle carries the Qwen3.5-4B files' template (enable_thinking defaults to false) and declares a `thought` channel (<think>\n / \n</think>); with thinking on, the CPU needs a budget — without one a 2,048-token turn did not close on the CPU, while thinking_token_budget 300 closed every thought in the measured runs (card L26, L52, L77).
- fp32 GPU activations are declared on purpose: with the runtime's fp16 default the int8 weights answered 0 of 8 check questions (every reply a run of `!`), at 7.8 % less time on the same workload (card L46, L96).
- One Engine per conversation: on a shared Engine the second conversation starts from the first one's recurrent state (LiteRT-LM issue 3165); the probe broke the second conversation in 2 of 3 int8 runs (every reply a repeated `</think>`), 3 of 3 runs were right on a fresh Engine (card L83).
- Tool calling: the template renders a tool list, the model writes the call in Qwen3.5's XML form as text, and the runtime (generic model type) does not parse or run it (LiteRT-LM issue 3927); a tool result sent back as a `tool` message is used in the answer (card L81).
- Tokenizer: the tokenizer section is the repo's tokenizer.json byte for byte; the exporter's class-written tokenizer (no `\p{M}` in the pre-tokenizer) had tokenized Hindi, Thai, Tamil and Arabic with harakat differently; the runtime now matches the tokenizers library in 346 of 346 strings (card L79).
- Init: litert-lm 0.18.0's CLI prints an init time that sums overlapping init phases, about twice the engine build in the log's timestamps; the GPU build is long because each of the 7 signature graphs compiles on one thread (34,373 nodes for the 1024-token prefill) (card L87).
- Galaxy S26: the int8 file was not started on the phone's GPU after a sibling 8B-class int8 file restarted this phone during its GPU engine warm-up (2026-10-14); the CPU was not run (by earlier CPU peak factors the file would need 13.8 GB against the phone's 11.66 GB) (card L100).
- Pixel 8a: not run; no Qwen3.5 Hub file built a GPU engine on this 7.75 GB Mali phone under a memory guard (card L101). iOS not measured: the file is one 8.57 GiB TFLite section, over the ~2 GiB single-section limit (card L42).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, fresh process per question, Mac GPU): bf16 97 / int8 98; the int8 file agrees with bf16 on 99 of 100 questions (card L33, L56-L64).
- Mac: the int8 file is the faster file on the CPU (prefill 228 vs 66 tok/s for the int4 file), the Mixed INT4 file on the GPU; without --cache, `run` writes a 7.99 GB compiled-weights cache next to the model; the benchmark and GSM8K ran on files whose TFLite section is byte-identical to this file's (the tokenizer section and metadata were replaced afterwards) (card L42, L64, L77).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
