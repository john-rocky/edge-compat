---
family: gpt-oss
license: apache-2.0
model_id: gpt-oss-120b-mixed-int4-ekv4096
source_url: https://huggingface.co/litert-community/gpt-oss-120b
task: text-generation
---

# gpt-oss-120b-mixed-int4-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | gpt-oss |
| **Source** | https://huggingface.co/litert-community/gpt-oss-120b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | the LiteRT Tensor API (the decoder written one tensor at a time from the checkpoint, one layer read at a time, no fp32 copy of the whole model), packed with litert-lm-builder 0.19.1; repro = john-rocky/hf-to-litertlm litert-lm-builder 0.19.1; LiteRT-LM 0.19.1 from PyPI for the Performance numbers; runtime floor LiteRT-LM 0.19.0 (card L19, L35, L53) |
| **Command** | `openai/gpt-oss-120b rev b5c939de -> each layer's 128 experts as one TFLite custom op `moe` (only the 4 routed experts computed for each token; the router with its bias, the top 4 and the softmax over their 4 logits as ordinary ops), the experts' biases carried by one more input block of 32 columns (exact), GELU in its tanh form in place of the source's clamped SwiGLU, attention with its biases, the per-head sink logits, the 128-token window on the even layers and YaRN RoPE as in the source; signatures prefill_128, prefill_1024 and decode, a 4,096-token KV cache; the source's chat template with four edits for LiteRT-LM; the harmony headers declared as channels (card L33-L43)` |
| **Quantization** | experts int4 blockwise (block 32, fp32 scales); attention q / k / v / o and lm_head int8 per channel; router, biases, sinks, norms and token table fp32; fp32 KV cache (card L29) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `gpt_oss_120b_mixed_int4_ekv4096.litertlm` | `13aa3c316a94579e51b2fd1d56beac89552025d31ef7d07d3e7a58028b11632c` | 73680.039 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-10/gpt-oss-120b-mixed-int4-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | pass | sig:the 12 check questions of line 3 (arithmetic, facts, translations, a rhyme, 0.9 vs 0.11, a word problem, 4 code tasks run on fixtures), greedy (top-k 1), at most 2,048 new tokens per answer, one conversation per question; verdicts on the final channel against the source model's replies (MLX) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-10 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:the card's example (the litert_lm Python API, CPU, 8 threads): one fixed question ('What is the capital of Japan? Answer briefly.') | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-10 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 49.94 | 4.75 | 20714.7 | 6978.268368 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-10 | measured |

## Pitfalls

- Memory: the 77.3 GB file is mapped into memory whole and the process adds 5.5 to 7.0 GB (peak resident memory 80.6 GB for the Python example, 82.1 GB for the benchmark); it runs on a Mac with 128 GB. The GPU has not been measured yet; no phone was tried (card L19, L60, L62).
- Not the source's exact computation: the experts' MXFP4 weights are re-quantized to int4, and the expert activation is GELU (tanh form) without the source's clamp at 7, because the moe op of LiteRT-LM 0.19.1 has no clamp and no SiLU. With that activation on the MXFP4 weights, the hidden state differs from the source's by 21 to 66 % (relative L2) from layer 6 on, and the next token is the source's at 399 of 413 reply positions (teacher-forced); this file adds the int4 experts on top. On 12 questions its greedy replies got the same verdict as the source model's on all 12 (card L23, L38, L70).
- Long replies: with LiteRT-LM 0.19.1 a blocking send_message still generating after 10 minutes fails with DEADLINE_EXCEEDED; the file decodes about 4.8 tokens/s on the M4 Max CPU, so 10 minutes is about 2,800 tokens, reasoning included; use send_message_async, or cap max_output_tokens (card L77).
- A message that does not fit: on LiteRT-LM 0.19.1, after a message fails because it does not fit in the context window (4,096 tokens), every later message in that conversation fails too; start a new conversation (card L78).
- Tool calls: the metadata type is generic_model, so LiteRT-LM returns a tool call as text: the function name in the commentary_header or tool_header channel, the arguments in the content, the reply stopping at <|call|>; to continue after a tool call, start a new conversation with the history (card L45).
- Sampling: the file carries Transformers' defaults for the source's generation_config (type TOP_P: top_k 50, top_p 1.0, temperature 1.0), so the text varies from run to run; the card's checks are greedy (card L46, L66, L101).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
