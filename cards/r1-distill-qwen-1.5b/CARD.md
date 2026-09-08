---
family: deepseek-r1
license: mit
model_id: r1-distill-qwen-1.5b
source_url: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
task: text-generation
---

# r1-distill-qwen-1.5b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | deepseek-r1 |
| **Source** | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | official upstream litert-torch export_hf (clean worktree at upstream/main, no fork); Qwen2ForCausalLM rides the stock converter, no custom code TODO (owner) — the card pins the tree ('upstream/main'), not a version number |
| **Command** | `TODO (owner) — not recorded in the source card` |
| **Quantization** | int4 weights — blockwise (block 32) + OCTAV optimal clipping, symmetric; embedding INT8; integer compute; KV cache 4096. File model.litertlm, ~1.0 GB |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | `69b35f01759eed765641ab4af589bbe98131fd2825662a086d9037409b8c1295` | 1748.516 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-22/r1-distill-qwen-1.5b__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-22 | measured |

## Pitfalls

- Reasoning model: opens a <think> ... </think> chain before the answer. The card's GSM8K run uses max_new_tokens=2048 — budget for the thinking block.
- GSM8K (n=100, greedy, 0-shot, identical prompt + extraction): int4 73.0% vs bf16 81.0% — at 1.5B, int4 costs ~8 pt (small-model 4-bit sensitivity; the 7B sibling is at -1 pt parity). Shipped as int4 for the best on-device size/speed.
- Prompt template is DeepSeek's own, bundled with the tokenizer: <｜User｜> / <｜Assistant｜>, stop token <｜end▁of▁sentence｜>.
- Gallery import: Google AI Edge Gallery 1.0.15+ supports .litertlm; v1.0.16+ can import directly from Hugging Face inside the app — the adb sideload steps are only needed on older builds or for local files.
- Measured decode ~116 tok/s (Mac M-series, Metal GPU, greedy); runs on 8 GB phones (iPhone / Android) at ~1 GB file size.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
