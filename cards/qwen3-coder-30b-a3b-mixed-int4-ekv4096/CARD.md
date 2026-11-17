---
family: qwen3-coder
license: apache-2.0
model_id: qwen3-coder-30b-a3b-mixed-int4-ekv4096
source_url: https://huggingface.co/litert-community/Qwen3-Coder-30B-A3B-Instruct
task: text-generation
---

# qwen3-coder-30b-a3b-mixed-int4-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3-coder |
| **Source** | https://huggingface.co/litert-community/Qwen3-Coder-30B-A3B-Instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | the LiteRT Tensor API (LiteRT dcb657e8 with a small patch in the conversion scripts), each layer's 128 experts written as one TFLite custom op moe; packed with litert-lm-builder 0.19.0; repro = john-rocky/hf-to-litertlm, (the pack, writer and bundle scripts are Qwen3-30B-A3B's) (card L21, L38-L39, L46) LiteRT dcb657e8 (+ a small patch) for the writer, litert-lm-builder 0.19.0; every number on the card from litert_lm_advanced_main built from LiteRT-LM dab2e708 with LiteRT dcb657e8 and ml-drift 76492bb9 plus the two open LiteRT pull requests #11192 and #11188 (2026-10-28); LiteRT-LM 0.19.1, the latest release on 2026-11-09, does not run the file (card L21, L25, L79-L99) |
| **Command** | `Qwen/Qwen3-Coder-30B-A3B-Instruct rev b2cff646 (bf16) -> the decoder written one tensor at a time, never as an fp32 graph: each layer's 128 experts in one moe custom op with activation silu (only the 8 routed experts computed per token; the router, its softmax, the top 8 and their renormalization as ordinary ops; int4 block scales below 2^-14 raised to 2^-14), the residual add after each moe written as (x + moe) + moe * 0 (--moe_fusion_guard); signatures prefill_128, prefill_1024 and decode; KV cache 4,096 tokens; the source's chat template rewritten for typed content parts, stop tokens <|im_end|> (151645) and <|endoftext|> (151643), the checkpoint's tokenizer.json byte for byte; generic_model with supports_function_calling and the tool_call_config of LiteRT-LM's models/qwen3_5 (card L38-L44, L75)` |
| **Quantization** | experts int4 blockwise (block 32, fp32 scales); attention q / k / v / o and lm_head int8 per channel; router, norms and token table fp32; fp32 KV cache of 4,096 tokens; fp32 activations declared (prefer_activation_type) (card L32, L34) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `qwen3_coder_30b_a3b_mixed_int4_ekv4096.litertlm` | `cda821eb926abf688db4acb84b08c653d5e2c55f9e7b84b2e835f2dea84d7908` | 19693.52 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-09/qwen3-coder-30b-a3b-mixed-int4-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | run_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |
| mac-studio-m4-max | gpu | load_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-09 | measured |

## Pitfalls

- No LiteRT-LM release runs this file: from PyPI, LiteRT-LM 0.19.1 stops on the CPU with `moe node #85 unsupported activation='silu'` and on the GPU with `moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`. SiLU in the moe op is in the LiteRT pull requests #11192 (GPU kernel) and #11188 (XNNPACK kernel, CPU), open on 2026-11-09; the card's numbers come from a LiteRT-LM built from source with both (card L25, L79-L99).
- On a LiteRT-LM built from source with #11192 and #11188 (not a release), the card measured this file on an Apple M4 Max (128 GB, macOS 27.0), 1,024 prefill and 256 decode tokens, 2026-11-09: WebGPU prefill 706.22 tok/s, decode 44.67 tok/s, peak physical footprint 34.04 GB; CPU (XNNPACK) 54.12 / 4.23 tok/s, 5.37 GB; on its 12 check questions and a 3-turn conversation, the same verdict as Transformers 12 / 12, the stop token 12 / 12, 3 turns pass on both backends (card L21, L52-L56, L65-L71).
- Keep the fp32 activation declaration: copies of the files that declare fp16 decoded 9 to 17 % faster, but no reply on WebGPU had an answer (empty, a single `*` before the stop token, or a short pattern repeated over the whole reply) (card L21, L41).
- The card's GPU numbers and checks used --sampler_handles_input=false; without it, two runs of the four code questions gave the same verdicts but not the same text. The Python, Kotlin and Swift APIs of LiteRT-LM 0.19.1 have no setting for it (card L113).
- Tool calls are XML (<tool_call>, <function=…>, <parameter=…>), declared through generic_model's tool_call_config; the card compared the calls' text with Transformers by giving the CLI the template's text as is (no template applied) and did not run the runtime's own parsing of a call (card L44).
- The sampler is TOP_P (top_p 0.8, top_k 20, temperature 0.7); the source's repetition_penalty of 1.05 is not in the file (LiteRT-LM's metadata has no field for it) (card L45).
- Size: 20.65 GB; no phone was tried (card L32, L61).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
