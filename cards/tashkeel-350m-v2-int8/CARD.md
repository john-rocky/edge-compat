---
family: granite
license: apache-2.0
model_id: tashkeel-350m-v2-int8
source_url: https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT
task: text-generation
---

# tashkeel-350m-v2-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/mlboydaisuke/Tashkeel-350M-v2-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm, family recipe (2026-08-25) (card L49-L50) TODO (the card names the converter but not its version) |
| **Command** | `convert_granite4h.py (hf-to-litertlm, 2026-08-25; invocation not stated in the card) (card L50)` |
| **Quantization** | dynamic int8 on linears + embedding (card L38) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Tashkeel-350M-v2_int8.litertlm` | `e962e6287a3b1e109498d1bca09d20fe55131bb1cd7431165bf255cfd2de1cd7` | 458.921 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/tashkeel-350m-v2-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/tashkeel-350m-v2-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/tashkeel-350m-v2-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 224 | - | 212.13 | 68.96 | 1070.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 224 | - | 754.69 | 46.26 | 320.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 761.25 | 97.21 | 346.7 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |
| pixel-8a | cpu | fallback | no | - | 224 | - | 111.8 | 49.06 | 2020.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| pixel-8a | gpu | pass | yes | - | 224 | - | 250.92 | 18.37 | 950.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Arabic diacritization (tashkeel) fine-tune of ibm-granite/granite-4.0-h-350m trained on Misraj/Sadeed_Tashkeela; the checkpoint's chat template (byte-equal to the granite base's, 6418/6418) is embedded and applied at runtime (card L19-L22, L52-L53).
- The metadata start token is dropped — granite's template has no leading BOS, and at 350M scale a prepended <|end_of_text|> flips correct diacritization into garbage (measured on this checkpoint) (card L54-L56).
- Task gate: the model card's worked example plus nine undiacritized MSA probes, bundle vs HF fp32 greedy on identical rendered strings: fp16 10/10 byte-identical, int8 8/10 (the card carries a note on the two int8 misses) (card L35-L45, L57-L58).
- One Engine per conversation: on litert-lm 0.18.0 a new Conversation on an Engine that already ran another conversation starts from the earlier conversation's recurrent state (it reuses the prompt tokens the two share and does not reset the state; LiteRT-LM issue #3165); in the card's probe (three questions in a pirate-persona conversation, then in one told to answer in one word, the same Engine against a fresh Engine in a new process; CPU, litert-lm 0.18.0 on a Mac) the _int8 file answered the capital question with an empty reply on the same Engine against 'The answer is Paris.' on a fresh one, in both of its runs; the test's questions are outside this model's task (Arabic diacritization), so it shows the Engine's behaviour, not the model's quality; create a new Engine for each conversation, the first conversation on an Engine is unaffected (card L59).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on these files' template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L97).
- License Apache-2.0, inherited from the source model and its granite base (card L101).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
