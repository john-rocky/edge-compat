---
family: qwen3
license: apache-2.0
model_id: qwen3-4b-thinking-dynamic-wi4b32-afp32
source_url: https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507
task: text-generation
---

# qwen3-4b-thinking-dynamic-wi4b32-afp32

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | LiteRT Torch (litert-torch) path, quantized with AI Edge Quantizer; the artifact incorporates LiteRT-LM GPU graph optimizations (composite ops for RoPE, fused QKV, …) (card L66) TODO (the card names the converter but not its version) |
| **Command** | `TODO (not stated in the card)` |
| **Quantization** | dynamic INT4 block-32 weights, FP32 activations (dynamic_wi4b32_afp32), KV 4096 (card L62, L66) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3_4b_thinking_dynamic_wi4b32_afp32.litertlm` | `110098cfd2766cc46a5138bb03b44f0350b7608052cb48321a47b9f0ebdb89d9` | 2168.84 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/qwen3-4b-thinking-dynamic-wi4b32-afp32__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/qwen3-4b-thinking-dynamic-wi4b32-afp32__raspberry-pi-5.json`, `data/device_runs/selfbuilt-b2f686e2/2026-10-07/qwen3-4b-thinking-dynamic-wi4b32-afp32__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | - | 23 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm selfbuilt-b2f686e2 · Android 16 | 2026-10-07 | measured |
| galaxy-s26 | gpu | pass | yes | - | 23 | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm selfbuilt-b2f686e2 · Android 16 | 2026-10-07 | measured |
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 129.44 | 11.2 | 1650.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 8.92 | 1.43 | 29403.6 | 4507.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The block-32 build is the CPU file: it degraded more on GSM8K (−9 pt vs bf16 90.0%, against −4 pt for the block-128 build) and produced corrupted output under the iPhone GPU delegate; the corruption is not iPhone-specific — on macOS Metal it degenerates too (question-echo loops, truncated reasoning, measured 2026-08-31 before and after the metadata repair). Use the block-32 file on CPU; block 128 is the recommended build on every backend (card L105-L117).
- Both bundles run fully delegated on the Galaxy S26 GPU per the card; this file: 3764/3764 ops across 3 subgraphs on LiteRT GPU, 1627 MB (card L115, L137); this repo carries the Pi 5 CPU row (device runs).
- 2026-08-31 metadata-only repairs (weights byte-identical, verified section by section): the thought channel (<think>\n / \n</think>) declared in the bundle metadata and the think pre-fill repaired — without the channel the runtime streams raw reasoning into the answer (card L68).
- litert-lm 0.18.0 passes message content to the chat template as a list of typed parts; the file published before 2026-10-08 (sha256 a412d77d…) stopped at the template on 0.18.0 with `tried to use + operator on unsupported types string and sequence` while answering on 0.17.1. The 2026-10-08 file (sha256 110098cf…, same size) carries a template that reads content as a string and as parts; only the template section changed, the graph, weights, tokenizer and executor metadata are byte-identical, and for string content it renders the same prompt byte for byte (card L45).
- Thinking cannot be switched off: on litert-lm 0.18.0 ThinkingConfig(enable_thinking=False) has no effect on this file (nor on model.litertlm): with a 400-token cap the thought channel uses the whole cap and no answer text appears (measured 2026-10-13); set thinking_token_budget (300 answered 17 × 23 under a 700-token cap) and a cap above it (card L76).
- Evaluating a reasoning model at a short token budget understates int4 — benchmark with max_tokens >= 2048 (card L72-L74, L108-L110).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on this file's template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L236).
- License Apache-2.0, inherited from Qwen/Qwen3-4B-Thinking-2507 (card L240-L241).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
