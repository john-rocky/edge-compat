---
family: smollm3
license: apache-2.0
model_id: smollm3-3b-litert
source_url: https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT
task: text-generation
---

# smollm3-3b-litert

| | |
|---|---|
| **Task** | text-generation |
| **Family** | smollm3 |
| **Source** | https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (generic path; SmolLM3ForCausalLM on the existing converter, the NoPE attention schedule — rotary disabled on every 4th layer — lowers to generic ops with no custom kernel) (card L112-L115) TODO (the card names the converter but not its version) |
| **Command** | `TODO (not stated in the card)` |
| **Quantization** | int4 blockwise (block 32) + OCTAV optimal-clipping, embedding INT8 externalized into its own section so the main weights section stays under the iOS ~2 GiB single-mmap limit; KV cache 4096 (card L29, L31, L115-L119) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `f9d25b1177c908041b67ea249ecb80cfc23b401810c735713035db73453ae757` | 1909.502 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.0/2026-09-06/smollm3-3b-litert__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |

## Pitfalls

- GSM8K (n=100, greedy, 0-shot CoT asking for '#### <n>', identical prompt and extraction): bf16 81.0% / LiteRT int4 (BOCTAV4) 81.0% — fully at parity; visible step-by-step chain-of-thought, clean stop at <|im_end|> (card L97-L108).
- Blockwise (not channelwise) int4 plus OCTAV is what holds reasoning accuracy at parity (card L118-L119).
- Template: on 2026-09-21 the file's chat template was updated to accept the 0.18 content-parts form (the string form unchanged); weights, tokenizer and executor metadata byte-identical (card L35).
- Thinking cannot be switched off: on litert-lm 0.18.0 ThinkingConfig(enable_thinking=False) has no effect on this file: with a 400-token cap the thought channel uses the whole cap and no answer text appears (measured 2026-10-13); set thinking_token_budget (300 answered 17 × 23 under a 700-token cap) and a cap above it (card L53).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on this file's template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L129).
- License Apache-2.0, inherited from HuggingFaceTB/SmolLM3-3B (card L133-L134).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
