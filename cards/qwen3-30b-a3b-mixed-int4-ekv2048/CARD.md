---
family: qwen3
license: apache-2.0
model_id: qwen3-30b-a3b-mixed-int4-ekv2048
source_url: https://huggingface.co/litert-community/Qwen3-30B-A3B
task: text-generation
---

# qwen3-30b-a3b-mixed-int4-ekv2048

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/Qwen3-30B-A3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | the LiteRT Tensor API (LiteRT dcb657e8 with a small patch in the conversion scripts), each layer's 128 experts written as one TFLite custom op moe; packed with litert-lm-builder 0.19.0; repro = john-rocky/hf-to-litertlm LiteRT dcb657e8 (+ a small patch) for the writer, litert-lm-builder 0.19.0; every number on the card from litert_lm_advanced_main built from LiteRT-LM dab2e708 with LiteRT dcb657e8 and ml-drift 76492bb9 plus the two open LiteRT pull requests #11192 and #11188 (2026-10-28); LiteRT-LM 0.19.0, the latest release on 2026-11-03, does not run the file (card L19, L23, L76-L96) |
| **Command** | `Qwen/Qwen3-30B-A3B rev ad44e777 (bf16) -> the decoder written one tensor at a time, never as an fp32 graph: each layer's 128 experts in one moe custom op with activation silu (only the 8 routed experts computed per token; the router, its softmax, the top 8 and their renormalization as ordinary ops; int4 block scales below 2^-14 raised to 2^-14), the residual add after each moe written as (x + moe) + moe * 0 (--moe_fusion_guard); signatures prefill_128, prefill_1024 and decode; KV cache 2,048 tokens; LiteRT-LM's own Qwen3 metadata verbatim (the chat template, stop token <|im_end|> (151645), the qwen3 model type, a thought channel), the checkpoint's tokenizer.json byte for byte (card L36-L40, L72)` |
| **Quantization** | experts int4 blockwise (block 32, fp32 scales); attention q / k / v / o and lm_head int8 per channel; router, norms and token table fp32; fp32 KV cache of 2,048 tokens; fp32 activations declared (card L29, L32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `qwen3_30b_a3b_mixed_int4_ekv2048.litertlm` | `a748069e80c09f377eb1cd4109a4e5aaae5f5658ada9da221c633dabf697c719` | 19693.676 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.0/2026-11-03/qwen3-30b-a3b-mixed-int4-ekv2048__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | run_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.0 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-03 | measured |
| mac-studio-m4-max | gpu | load_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.0 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.0 · macOS 27.0 (26A428) | 2026-11-03 | measured |

## Pitfalls

- No LiteRT-LM release ran this file on the card's date: from PyPI, LiteRT-LM 0.19.0 stops on the CPU with `moe node #85 unsupported activation='silu'` and on the GPU with `moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`. SiLU in the moe op is in the LiteRT pull requests #11192 (GPU kernel) and #11188 (XNNPACK kernel, CPU), open on 2026-11-03; the card's numbers come from a LiteRT-LM built from source with both (card L23, L76-L96).
- On a LiteRT-LM built from source with #11192 and #11188 (not a release), the card measured this file on an Apple M4 Max (128 GB, macOS 27.0) with WebGPU, 1,024 prefill and 256 decode tokens, 2026-11-03: prefill 1,043.42 tok/s, decode 54.81 tok/s, peak physical footprint 32.75 GB; on its 12 check questions and a 3-turn conversation, the same verdict as Transformers 12 / 12, the stop token 12 / 12, 3 turns pass; the KV cache holds 2,048 tokens (card L19, L32, L47-L51, L62-L66).
- Keep the fp32 activation declaration: with fp16 activations every one of the 8 check questions run repeated a short passage of its thought until the token limit, without an answer (card L41).
- Every GPU check on the card ran with --sampler_handles_input=false; the default GPU sampler path was compared on one 64-token reply only (card L70).
- Thinking is on by default: the CLI prints the thought between [thought] and [/thought], then the answer; the sampler is TOP_P (top_p 0.95, top_k 20, temperature 0.6) (card L21, L40, L108).
- Int4 attention and lm_head weights were not faster than int8 ones on WebGPU (41.9 against 41.1 tokens/s) and one of the 12 check questions looped with them, so the files keep them int8 (card L42).
- Size: 20.65 GB; no phone was tried (card L29, L58).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
