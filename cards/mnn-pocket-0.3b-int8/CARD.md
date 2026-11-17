---
family: mnn-pocket
license: apache-2.0
model_id: mnn-pocket-0.3b-int8
source_url: https://huggingface.co/litert-community/MNN-Pocket-0.3B
task: text-generation
---

# mnn-pocket-0.3b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | mnn-pocket |
| **Source** | https://huggingface.co/litert-community/MNN-Pocket-0.3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 export_hf (fused QKV and gate/up projections, GPU composites, the attention composites) packed with litert-lm-builder 0.18.1; repro = hf-to-litertlm REPRODUCE.md, entry "2026-11-07 update" (card L195, L201) litert-torch 0.10.0, litert-lm-builder 0.18.1 (litert-torch 0.10.0 calls a builder API that litert-lm-builder 0.19 no longer has); LiteRT-LM 0.19.1 for every measurement (card L194-L195) |
| **Command** | `taobao-mnn/MNN-Pocket-0.3B rev 087a52fd -> export_hf --use_qkv_norm_rope_composite=True --use_sdpa_composite=True, prefill signatures 128 and 1,024, --cache_length 4096, recipe dynamic_wi8_afp32; the source's tokenizer.json unchanged, the source's chat template with two changes, stop tokens 65512 and 65510, model type generic_model (card L193, L195, L198-L200)` |
| **Quantization** | int8 weights (one scale per output channel) with float activations, fp16 on the GPU; KV cache of 4,096 tokens; prefill signatures of 128 and 1,024 tokens; the input embedding and the output layer share one int8 table (card L31-L32, L41) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `MNN-Pocket-0.3B_int8.litertlm` | `092a838999e99f4ab03239ca8de1c0caba01d1772d7a22909e76988b0b3dc1e9` | 305.2 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-07/mnn-pocket-0.3b-int8__galaxy-s26.json`, `data/device_runs/0.19.1/2026-11-07/mnn-pocket-0.3b-int8__mac-studio-m4-max.json`, `data/device_runs/0.19.1/2026-11-07/mnn-pocket-0.3b-int8__pixel-11-pro.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | pass | sig:the device line on the CPU: 8 short questions (each must end before the 1,024-token cap) and the first 20 GSM8K test questions (4-shot chain of thought, greedy, cap 1,024 new tokens), every reply compared with the Mac CPU's reply at 1 thread | - | - | - | - | 949.4569 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006) | 2026-11-07 | measured |
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 333.3525 | 53.242 | 3157.0 | 2432.8479 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006) | 2026-11-06 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 5959.679 | 105.5755 | 181.0 | 374.53 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU) · litert-lm 0.19.1 · Android 16 (BP4A.251205.006) | 2026-11-06 | measured |
| mac-studio-m4-max | cpu | pass | - | pass | sig:GSM8K-100 (the test split's first 100, 4-shot chain of thought, greedy, cap 1,024 new tokens), three fresh processes; bar 53 = the source model in bfloat16 (58) minus 5 | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-07 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 608.465 | 95.85 | 1813.1 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-06 | measured |
| mac-studio-m4-max | gpu | pass | - | pass | sig:GSM8K-100 (the test split's first 100, 4-shot chain of thought, greedy, cap 1,024 new tokens), three fresh processes; bar 53 = the source model in bfloat16 (58) minus 5 | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-07 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 18232.525 | 490.505 | 58.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-06 | measured |
| pixel-11-pro | cpu | fallback | no | pass | sig:the device line on the CPU: 8 short questions (each must end before the 1,024-token cap) and the first 20 GSM8K test questions (4-shot chain of thought, greedy, cap 1,024 new tokens), every reply compared with the Mac CPU's reply at 1 thread | - | - | - | - | 866.2753 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.1 · Android 17 (CD1A.260905.001.B1) | 2026-11-07 | measured |
| pixel-11-pro | cpu | fallback | no | - | 1024 | - | 162.566 | 45.418 | 6483.5 | 2158.5183 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.1 · Android 17 (CD1A.260905.001.B1) | 2026-11-07 | measured |
| pixel-11-pro | gpu | pass | yes | - | 1024 | - | 2781.3865 | 28.272 | 403.5 | 292.5568 | Pixel 11 Pro · Tensor G6, Imagination PowerVR GPU · litert-lm 0.19.1 · Android 17 (CD1A.260905.001.B1) | 2026-11-07 | measured |

## Pitfalls

- Tool calls come back as text: given tool definitions, the model writes the call as a bare JSON object without the `<tool_call>` tags its template describes (the source model in transformers does the same on the card's test prompt), and the file carries no tool-call configuration, so parse the JSON in your app (card L183).
- GPU needs LiteRT-LM 0.19.0 or newer, the first release whose GPU kernels run the attention composites: with 0.18.1 the Mac GPU reports a shader error and the reply is a string of symbols, while its CPU runs them (card L185).
- Context is 4,096 tokens, prompt and reply together (the source model supports 32K in deployment). The 2,048-token sibling decodes faster everywhere but on the Mac GPU, where the two tie; take this file when a conversation needs more than 2,048 tokens (card L37, L186).
- Phone CPUs: this file decodes 2 % (Galaxy S26) and 7 % (Pixel 11 Pro) slower than the first upload's file, re-measured in the same session, and its peak memory is about 0.76 GB higher (card L37). Both phones lowered their CPU clocks within the first run of every CPU process; on the Pixel 11 Pro the third run of every 4,096-token CPU process decoded 26 to 40 % slower than the second, so the four runs behind the 45.4 tokens/s cell span 35 to 59 tokens/s (card L134).
- Model type: the file declares generic_model. LiteRT-LM 0.19.1 gives any other type different GPU settings (allow_src_quantized_fc_conv_ops is on for every type except generic_model). The first upload, typed qwen2p5, failed the answer checks on the Galaxy S26 GPU; a copy with only the type changed passed on both phone GPUs; how much the type itself makes was not measured (card L198).
- Phone GPU checks: this file's GPU checks on the phones ran on a build with the same four sections, not on these bytes (card L65, L175).
- Chat template: LiteRT-LM hands each message's content to the template as a list of text parts and the source template keeps only string content (every user message rendered empty), and the source template errors on an empty message list; the bundled template reads both forms and renders the empty list, giving transformers' text byte for byte on 7 test conversations (card L199).
- Sampling: the file carries the source card's example settings (top-k 20, top-p 0.8, temperature 0.8); every check on the card is greedy (card L41, L160).
- No int4 file: none of the int4 builds passed the Mac checks on both the CPU and the GPU, including one with the authors' quantization-aware int4 values (card L184).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
