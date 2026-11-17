---
family: kimi-linear
license: mit
model_id: kimi-linear-48b-a3b-mixed-int8-cpu-ekv4096
source_url: https://huggingface.co/litert-community/Kimi-Linear-48B-A3B-Instruct
task: text-generation
---

# kimi-linear-48b-a3b-mixed-int8-cpu-ekv4096

| | |
|---|---|
| **Task** | text-generation |
| **Family** | kimi-linear |
| **Source** | https://huggingface.co/litert-community/Kimi-Linear-48B-A3B-Instruct |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | a graph writer for litert-torch (main 67f11e8): plain TFLite ops plus LiteRT's moe custom op, checked against Transformers' kimi_linear model; repro = john-rocky/hf-to-litertlm, at e945a06 (card L19) litert-torch main 67f11e8 for the writer; every number on the card from litert_lm_advanced_main built from LiteRT-LM main 02d7e8d8 with LiteRT bf49eb62 and the two open LiteRT pull requests #11192 and #11188; LiteRT-LM 0.19.2, the latest release on PyPI on 2026-11-16, does not run the file, and neither does 0.19.1 (card L19, L21) |
| **Command** | `TODO (the card gives no single conversion command; the scripts and the check of each part are in john-rocky/hf-to-litertlm kimi_linear_48b_a3b_work at e945a06). Source moonshotai/Kimi-Linear-48B-A3B-Instruct at revision e1df551a, bf16 weights; each MoE layer's 256 routed experts in one moe op with SiLU, the router, the shared expert and the dense first layer in plain ops; signatures prefill_128, prefill_1024 and decode, context 4096 tokens; a tokenizer.json converted from the source's tiktoken tokenizer; the source's chat template with four rewrites for LiteRT-LM's template engine` |
| **Quantization** | the routed experts int8 per channel (one scale per output row and expert); every other weight fp32, the token table and the output layer included; fp32 activations; context 4096 tokens; 55.29 GB (card L28) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `kimi_linear_48b_a3b_mixed_int8_cpu_ekv4096.litertlm` | `18a9fd55228e650c9229ffdc69207b493583065a525f614c92ecd0659afb76ca` | 52724.755 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.19.1/2026-11-13/kimi-linear-48b-a3b-mixed-int8-cpu-ekv4096__mac-studio-m4-max.json`, `data/device_runs/0.19.2/2026-11-13/kimi-linear-48b-a3b-mixed-int8-cpu-ekv4096__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | run_failed | - | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-13 | measured |
| mac-studio-m4-max | cpu | run_failed | - | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.2 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.2 · macOS 27.0 (26A428) | 2026-11-13 | measured |
| mac-studio-m4-max | gpu | load_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.1 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.1 · macOS 27.0 (26A428) | 2026-11-13 | measured |
| mac-studio-m4-max | gpu | load_failed | no | - | sig:one fixed question ('What is 17 + 25? Answer briefly.') through the PyPI litert-lm 0.19.2 CLI (`litert-lm run`, top-k 1), the released runtime's loadability check | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.19.2 · macOS 27.0 (26A428) | 2026-11-13 | measured |

## Pitfalls

- No LiteRT-LM release runs this file: LiteRT-LM 0.19.2 (the latest release on PyPI on 2026-11-16) and 0.19.1 stop on the CPU with `moe node #708 unsupported activation='silu'` and on the GPU with `moe only supports activation='gelu_tanh' (or its legacy spelling 'gelu')`, and print `An error occurred`; SiLU is in the two open LiteRT pull requests #11192 and #11188 (card L21).
- On a LiteRT-LM built from source with #11192 and #11188 (not a release), the card measured this file on an Apple M4 Max (128 GB), 1,024 prefill and 256 decode tokens (2026-11-13): CPU (XNNPACK, 12 threads) prefill 134.09 tok/s, decode 9.72 tok/s, peak footprint 19 GB; on its 12 questions and a 3-turn conversation, the same verdict as Transformers 12 / 12, the stop token 12 / 12 and the 3 turns pass (card L19, L38, L40, L50, L52).
- Which file: the repo publishes this file for the CPU only; on the same Mac the int8 file decodes faster on the CPU too (11.98 against 9.72 tok/s) and is the one for the GPU (card L27-L28, L37-L38, L42).
- The file is computed in fp32 on the CPU: its only int8 weights are the experts', which the CPU moe kernel turns back into fp32 before it uses them. The card's CPU command sets 12 threads (--num_cpu_threads=12); without the flag the runtime uses 4 (card L54, L66-L72).
- Once a message failed because it did not fit in the context window, every later message in that conversation failed as well (the next one with `Task failed with state: 7`): start a new conversation after such an error, on a new engine (card L79).
- One conversation per engine: the Kimi Delta Attention layers carry a recurrent state, so create a new engine for each conversation (LiteRT-LM #3165) (card L78).
- The file carries no sampler settings of its own; the card's checks decode greedily (--sampler_top_k=1).
- Context 4096 tokens (the source supports up to 1M) (card L80).
- Measured on a desktop only (Apple M4 Max, 128 GB): the file (55.29 GB) does not fit in a phone's memory, and Android and iOS were not measured (card L81).
- Tool calls were not checked: the file declares LiteRT-LM's generic model type with no tool-call configuration, so LiteRT-LM returns a tool call as reply text (card L82).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
