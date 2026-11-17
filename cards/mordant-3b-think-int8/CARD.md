---
family: granite
license: apache-2.0
model_id: mordant-3b-think-int8
source_url: https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT
task: text-generation
---

# mordant-3b-think-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/mlboydaisuke/Mordant-3B-Think-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | hf-to-litertlm, one command (python scripts/convert.py Kezmark/Mordant-3B-Think, 2026-08-25) (card L35-L36) TODO (the card names the converter but not its version) |
| **Command** | `python scripts/convert.py Kezmark/Mordant-3B-Think (hf-to-litertlm, 2026-08-25) (card L36)` |
| **Quantization** | dynamic int8 on linears + embedding (card L29) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Mordant-3B-Think_int8.litertlm` | `315776fa9c1e4ae579f4632f0b4a83d185e80964b72dcace727f1ee724dc32f6` | 3587.91 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/mordant-3b-think-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/mordant-3b-think-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 206 | - | 62.91 | 7.7 | 3400.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 206 | - | 232.68 | 9.59 | 990.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 97.25 | 20.18 | 2682.0 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1128.98 | 71.85 | 240.7 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-25 | measured |

## Pitfalls

- A full fine-tune of ibm-granite/granite-4.1-3b for AI image-generation prompt composition with chain-of-thought reasoning; the finetune's thinking-form chat template (opens the assistant turn with <think>) is embedded verbatim (byte-equal 1474/1474) (card L19-L21, L38-L40).
- Template: on 2026-09-21 the file's chat template was updated to accept the 0.18 content-parts form (the string form unchanged); weights, tokenizer and executor metadata byte-identical (card L31).
- The spurious metadata start token is dropped: this family declares bos == eos == <|end_of_text|> and its template never renders a leading BOS, so an engine-prepended start token reads as 'this document already ended' — measured on this checkpoint it flips HF bf16 greedy output into a code-fence loop (card L41-L44).
- Reduced set of 7 prefill signatures + externalized embedder (the >=3B ship shape; the full set of 11 signatures is killed by iOS at Metal init on this family); quality check 8/8 (think-aware budget), non-degenerate (card L45-L47).
- Thinking cannot be switched off: on litert-lm 0.18.0 ThinkingConfig(enable_thinking=False) has no effect on this file: with a 400-token cap the thought channel uses the whole cap and no answer text appears (measured 2026-10-13); set thinking_token_budget (300 answered 17 × 23 under a 700-token cap) and a cap above it (card L68).
- Second turn of a conversation: the chat template opens a thought in every answer, so after each answer the runtime rebuilds the cache from the re-rendered history at the next turn; the second turn is free of the defect when the conversation opens with a user message, and the same source shows a zero embedding at another token from the third turn on (not measured). The defect: after an answer without a thought (no <think> … </think> block), from the second turn on LiteRT-LM puts the previous answer's stop token into the KV cache with a zero embedding, because the file holds the embedding table in its own section and the decoder takes embeddings, so later answers can differ from the answer to the same messages in one prompt (read in the source of LiteRT-LM 0.16.0 to 0.18.1; measured with 0.18.1 on WeVisOne-VL-2B and Qwen2.5-Coder-1.5B-Instruct, not on this file); one question per conversation, or a new conversation that starts with the earlier messages and the new question, avoids both (card L72).
- Final mark on LiteRT-LM before 0.17.0 (source read 0.10.1 to 0.16.1): an answer whose last punctuation mark is a token of its own, such as a . or ? after a word, arrives without that mark: the file carries stop sequences of <|end_of_text|> and a newline with each of 13 characters in front (12 punctuation marks and a space), so the runtime holds the mark back, the answer ends on <|end_of_text|> alone and the held mark is never written; 0.17.0 and later write the held mark (StopTokenStreamFilter in runtime/core/tasks.cc, source read 0.17.0 to 0.18.1); not measured on this file: it follows from the source read with the file's stop list and tokenizer, and the drop itself was seen with LiteRT-LM 0.16.1 on a Galaxy S26, on another bundle whose stop sequences have the same form (card L74).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on this file's template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L76).
- License Apache-2.0, inherited from the source model and its granite-4.1 base (card L80).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
