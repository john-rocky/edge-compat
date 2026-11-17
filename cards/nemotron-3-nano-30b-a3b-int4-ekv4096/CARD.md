---
family: nemotron-3-nano
license: other (nvidia-nemotron-open-model-license)
model_id: nemotron-3-nano-30b-a3b-int4-ekv4096
source_url: https://huggingface.co/litert-community/Nemotron-3-Nano-30B-A3B
task: text-generation
---

# nemotron-3-nano-30b-a3b-int4-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nemotron-3-nano |
| **Source** | https://huggingface.co/litert-community/Nemotron-3-Nano-30B-A3B |
| **License** | other (nvidia-nemotron-open-model-license) |

## Conversion

| | |
|---|---|
| **Tool** | TODO (the card names no converter). The graph, as the card states it: each layer's experts are one `moe` custom op; relu² is written as its `gelu_tanh` (the activation was checked through the logits); each `moe` output has a second consumer (`moe * 0`) so ML Drift does not fuse the next op into it (card L40) TODO (the converter's version is not stated); runtime LiteRT-LM 0.19.1: the CPU through the PyPI package, the GPU through the v0.19.1 command-line tool built from source (7a03724c) (card L27, L76-L80) |
| **Command** | `TODO (the card gives no conversion command); source nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, quantized (card L13-L14, L27)` |
| **Quantization** | experts int4 in blocks of 128 values (64 in their down projection); the other weight matrices int4, except the token table and lm_head (int8); router fp32; fp32 activations declared; a 4,096-token context; 17.8 GB (card L36, L40) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `nemotron3_nano_30b_a3b_int4_ekv4096.litertlm` | `cf6aa824ae6623355ffc82b4d844706517fdd61887f6a72cc9e5e9f32f449282` | 17017.559 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-12/nemotron-3-nano-30b-a3b-int4-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | pass | sig:the 12 check questions of line 3 (the 8 short questions of hf-to-litertlm's scripts/verify_quality.py, a price word problem, a syllogism, an average speed, a question in Japanese) and its 3-turn conversation, thinking off (enable_thinking false), greedy (top-k 1), at most 2,048 new tokens, one process per question; verdicts against the source model's replies (float32 on the bf16 weights, thinking off) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | cpu | pass | - | pass | sig:the 12 check questions of line 3 (the 8 short questions of hf-to-litertlm's scripts/verify_quality.py, a price word problem, a syllogism, an average speed, a question in Japanese) and its 3-turn conversation, thinking on (the template's default), greedy (top-k 1), at most 2,048 new tokens, one process per question; verdicts by the line's regex on the text after the thought | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 105.99 | 12.16 | 9740.0 | 4775.69 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the 12 check questions of line 3 (the 8 short questions of hf-to-litertlm's scripts/verify_quality.py, a price word problem, a syllogism, an average speed, a question in Japanese) and its 3-turn conversation, thinking off (enable_thinking false), greedy (top-k 1), at most 2,048 new tokens, one process per question; verdicts against the source model's replies (float32 on the bf16 weights, thinking off) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the 12 check questions of line 3 (the 8 short questions of hf-to-litertlm's scripts/verify_quality.py, a price word problem, a syllogism, an average speed, a question in Japanese) and its 3-turn conversation, thinking on (the template's default), greedy (top-k 1), at most 2,048 new tokens, one process per question; verdicts by the line's regex on the text after the thought | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 815.32 | 70.657 | 1270.0 | 19480.72 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-12 | measured |

## Pitfalls

- Keep the GPU cache off (cache_dir=":nocache" in Python, --disable_cache=true in the command-line tool): on a Mac, runs that loaded an earlier run's GPU program cache answered wrongly (card L68).
- One Engine per conversation: the Mamba-2 state is not reset between conversations (LiteRT-LM #3165) (card L69).
- Tool calls come back as text (the model's <tool_call> form) (card L70).
- Long replies: a blocking send_message still generating after 10 minutes fails with DEADLINE_EXCEEDED; use send_message_async or cap max_output_tokens (card L71).
- After a message overflows the context window, every later message in that conversation fails (Task failed with state: 7); start a new conversation (card L72).
- GPU: the card runs it with the v0.19.1 command-line tool built from source, which has --sampler_handles_input=false (the PyPI package does not); LiteRT-LM 0.18.0's GPU moe op only supports activation 'gelu'. CPU: the PyPI package, litert-lm 0.19.1 (card L27, L76-L93).
- Thinking is on by default; thinking_config=litert_lm.ThinkingConfig(enable_thinking=False) in create_conversation turns it off (card L29, L105).
- The card's checks (12 questions and a three-turn chat, a name then 17 × 23, greedy, against NVIDIA's bf16 weights): with thinking on, 12/12 right and 391 for 17 × 23 on all four files on the CPU and the GPU (the source model was not run with thinking on); with thinking off, 11/12 right, as the source model, and 17 × 23 comes back as 391 on the CPU and 411 on the GPU for the int4 files (this one among them), 401 for the mixed_int4 files and for the source model (card L59-L64).
- Which file: this one gives faster decode than mixed_int4 ekv4096 (card L36). On an Apple M4 Max (128 GB) its benchmark peaked at 20.4 GB on the GPU and 5.0 GB on the CPU (2026-11-12); speed still being tuned (updates follow) (card L27, L48-L49, L55).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
