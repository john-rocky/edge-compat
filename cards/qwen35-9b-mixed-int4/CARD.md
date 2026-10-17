---
family: qwen3.5
license: apache-2.0
model_id: qwen35-9b-mixed-int4
source_url: https://huggingface.co/litert-community/Qwen3.5-9B
task: text-generation
---

# qwen35-9b-mixed-int4

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
| **Quantization** | Mixed INT4: int4 blockwise-32 min-max on the linear layers, int8 on the lm_head and the embedding, the 48 decay/gate projections float; 216,006,656 block scales, all normal floats (none zero or subnormal); fp32 GPU activations declared (card L31, L108) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-9B_mixed_int4.litertlm` | `61042ebd3c473763972d22310c055f63fab5a29bc2976ab533d6a59c262cac65` | 5779.607 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-16/qwen35-9b-mixed-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-16/qwen35-9b-mixed-int4__mac-studio-m4-max.json`.

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

- Text only: the decoder of the multimodal release; no image or video input. Only the 8 attention layers keep a growing KV cache (a 4096-token budget); the 24 linear-attention layers carry a constant-size conv and recurrent state (card L24).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (Mac CPU, one run) (card L20).
- Thinking is off unless asked (the Qwen3.5-4B files' template, a `thought` channel). With thinking on and no budget a 12-pencil word problem kept this file's thought open to the 2,048-token cap in 3 of 3 CPU runs and a three-turn conversation left turns empty at 1,024 tokens per turn on both backends; with thinking_token_budget 300 every measured thought closed (card L52, L77).
- Galaxy S26 GPU: the engine of this file does not build on a phone of this memory class: on a device-lab Galaxy S26 the session ended during the engine build after the delegate had taken two prefill signatures, with the low-memory killer removing other apps, and the same-architecture Ornith-1.0-9B files failed four sessions the same way, a three-signature variant included; the CPU was not run (12.1 GB by earlier CPU peak factors against 11.66 GB) (card L96, L100).
- fp32 GPU activations are declared (the int8 sibling answered 0 of 8 with the runtime's fp16 default) (card L46, L96).
- One Engine per conversation: on a shared Engine the second conversation broke in 1 of 3 runs of this file (every reply a repeated `</think>`); LiteRT-LM issue 3165 (card L83).
- Tool calling: the model writes the call in Qwen3.5's XML form as text; the runtime does not parse or run it (LiteRT-LM issue 3927) (card L81).
- Tokenizer: the tokenizer section is the repo's tokenizer.json byte for byte (346 of 346 strings match the tokenizers library) (card L79).
- Init: the 0.18.0 CLI's printed init sums overlapping init phases — this file's GPU run printed 89.2 s for a 44.5 s engine build; each of the 7 signature graphs compiles on one thread (card L87).
- Pixel 8a: not run (no Qwen3.5 Hub file built a GPU engine on this Mali phone under a memory guard) (card L101). iOS not measured: one 5.64 GiB TFLite section, over the ~2 GiB single-section limit (card L42).
- Accuracy: GSM8K n=100 (thinking off, greedy, at most 2,048 output tokens, Mac GPU): bf16 97 / Mixed INT4 95; three of its replies (questions 38, 41, 62) reached the 2,048-token cap without a final line and count as wrong; the file is 34 % smaller than the int8 file (card L33, L58-L64).
- Mac: the Mixed INT4 file is the faster file on the GPU (13.9 % less time, decode 46.1 vs 38.4 tok/s) and the slower on the CPU (blockwise prefill 66 tok/s against 228); without --cache, `run` writes a 4.96 GB compiled-weights cache next to the model; the benchmark and GSM8K ran on files whose TFLite section is byte-identical to this file's (card L42, L64, L77).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
