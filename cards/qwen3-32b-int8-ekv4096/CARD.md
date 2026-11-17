---
family: qwen3
license: apache-2.0
model_id: qwen3-32b-int8-ekv4096
source_url: https://huggingface.co/litert-community/Qwen3-32B
task: text-generation
---

# qwen3-32b-int8-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-32B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 export_hf with GPU composites (one prefill and one decode signature), quantized with ai-edge-quantizer 0.10.1; repro = john-rocky/hf-to-litertlm litert-torch 0.10.0, ai-edge-quantizer 0.10.1; LiteRT-LM 0.19.1 built from its tag (7a03724c) for every number on the card, the PyPI litert-lm 0.19.1 for one question on the GPU (card L19, L23, L71) |
| **Command** | `Qwen/Qwen3-32B rev 9216db57 (bf16) -> export_hf with GPU composites, one prefill and one decode signature whose lengths LiteRT-LM sets at load time (prefill 1,024 and KV 4,096 by default); int8 per channel for every linear layer, lm_head and token table; LiteRT-LM 0.19.1's own Qwen3 metadata byte for byte (chat template with enable_thinking, a thought channel, stop token <|im_end|>, sampler top-k 20 / top-p 0.95 / temperature 0.6); the checkpoint's tokenizer.json byte for byte (card L29, L33, L35)` |
| **Quantization** | int8 per channel for every linear layer, lm_head and token table; fp32 activations declared; for the GPU backend (card L19, L29) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `qwen3_32b_int8_ekv4096.litertlm` | `c4d505aa0339e4e9fe5a6974b6e915e2dcfd2c441f2c865b6518df06b4b6008a` | 31368.284 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-09/qwen3-32b-int8-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 50.41 | 5.03 | 20510.0 | 5557.46 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:the 12 check questions of line 3 (arithmetic, facts, two translations, a rhyme, 0.9 vs 0.11, a word problem, fib(10) in Python, the planets in order) and a 3-turn conversation, greedy (--sampler_top_k=1), thinking on, at most 2,048 new tokens per answer, one process per question; verdicts against Transformers (bf16) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 160.537 | 11.86 | 6463.3 | 34295.6 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |

## Pitfalls

- Use the GPU: on LiteRT-LM 0.19.1's CPU backend the file runs, but the runtime quantizes each int8 layer's input to int8, and the file's top-1 token matched the bf16 checkpoint's at 18 of 24 positions of the test prompt (3 runs, identical), one miss at a 5.1-logit margin; without it, 23 of 24. The CPU was not checked on the 12 questions (card L23, L63).
- Long replies: with LiteRT-LM 0.19.1 a blocking send_message still generating after 10 minutes fails with DEADLINE_EXCEEDED; the M4 Max GPU decodes about 12 tokens/s, so on a slower machine a long thought can hit that limit; use send_message_async, or cap max_output_tokens (card L64).
- A message that does not fit: on LiteRT-LM 0.18.1 and 0.19.1, after a message fails because it does not fit in the context window, every later message in that conversation fails too (Task failed with state: 7); start a new conversation (card L65).
- Size: the file is 32.9 GB; the Apple M4 Max (128 GB) GPU benchmark peaked at a 35.97 GB physical footprint (vmmap); no phone was tried (card L29, L44, L48-L49).
- The card's numbers come from litert_lm_advanced_main built from the v0.19.1 tag (its command-line tool has the benchmark and sampler flags), with --sampler_handles_input=false on the GPU; without --sampler_top_k=1 the CLI samples with the file's TOP_P settings (card L23, L46, L67-L76, L82, L86).
- Thinking is on by default; the source card advises against greedy decoding in thinking mode (endless repetitions); the card's checks are greedy only to compare token by token (card L21, L53).
- No int4 file: an int4 version of the same export (block 128, min-max) failed the accuracy check (on the GPU, 3 of 12 questions and one of 3 turns ran to the 2,048-token cap; on the CPU, 12 of 24 next tokens matched the bf16 checkpoint's) (card L34).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
