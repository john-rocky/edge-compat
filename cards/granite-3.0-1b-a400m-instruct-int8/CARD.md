---
family: granite
license: apache-2.0
model_id: granite-3.0-1b-a400m-instruct-int8
source_url: https://huggingface.co/litert-community/granite-3.0-1b-a400m-instruct
task: text-generation
---

# granite-3.0-1b-a400m-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-3.0-1b-a400m-instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch at git commit 0d3869c, `litert_moe_sequential` route with each layer's 32 expert FCs fused into two FCs; repro = hf-to-litertlm granitemoe_work/convert_granite30_1b_a400m.sh (card L113) litert-torch 0d3869c, transformers 5.20.1, torch 2.14.0, ai-edge-quantizer-nightly 0.10.0.dev20261019, litert-converter 0.6.0.dev20261016, ai-edge-litert-nightly 2.4.0.dev20261018; litert-lm 0.18.0 for the activation-type step (card L113) |
| **Command** | `ibm-granite/granite-3.0-1b-a400m-instruct rev ffec3c35 -> export with dynamic_wi8_afp32 (prefill signatures 128 and 1024 plus decode over a 2048-position KV cache; the checkpoint's tokenizer.json packed byte for byte; IBM's template with three additions for litert-lm 0.18; PYTHONHASHSEED=0), the experts as two FCs per layer (gate_up [32768, 1024], down [1024, 16384], each expert's routing weight multiplied into its slice), the decode logits padded to 49,156 columns with one -1e4 column -> scripts/set_activation_type.py --type fp32 (prefer_activation_type on the TFLite section; every section's bytes unchanged) (card L113-L117)` |
| **Quantization** | int8 dynamic on linears, float activations; each layer's 32 experts as two fully connected layers; fp32 GPU activations declared (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-3.0-1b-a400m-instruct_int8.litertlm` | `b0b786a5672c25d4aeb8ed90b7363294703809a1f95a5dfb8fb3dcf5db11d39f` | 1309.152 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-21/granite-3.0-1b-a400m-instruct-int8__galaxy-s26.json`, `data/device_runs/0.18.0/2026-10-21/granite-3.0-1b-a400m-instruct-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-21/granite-3.0-1b-a400m-instruct-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 361.6425 | 23.255 | 2885.0 | 2103.054 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-21 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 351.305 | 27.285 | 2955.0 | 539.12 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.18.0 · Android 16 (BP4A.251205.006) | 2026-10-21 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 617.87 | 53.05 | 1676.1 | 4700.69 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-21 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 3236.885 | 146.34 | 323.2 | 2548.714 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-21 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 130.85 | 12.8175 | 7945.0 | 2071.757 | Pixel 8a · Google Tensor G3 · litert-lm 0.18.0 · Android 16 (CP1A.260505.005) | 2026-10-21 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 128.35 | 12.5825 | 8060.0 | 1363.497 | Pixel 8a · Google Tensor G3 · litert-lm 0.18.0 · Android 16 (CP1A.260505.005) | 2026-10-21 | measured |

## Pitfalls

- Dense cost: litert-torch's sparse litert_moe export computes the experts with GELU-tanh while this model's experts use SiLU, and LiteRT-LM 0.18.0's GPU refuses the moe op it writes, so the file takes the litert_moe_sequential route: all 32 experts of a layer for every token with the model's SiLU, a token costs the compute of all 1.33B parameters, not 400M (card L27).
- GPU activations: the file declares fp32 GPU activations. With the runtime's default fp16 activations the same weights scored GSM8K 28 / 29 / 29 on the Mac GPU against 35 / 35 / 35 with the declaration (bf16 34); the declaration costs the Mac GPU 5.4 % in decode and 17.7 % in prefill (card L50-L57).
- Phones' GPUs: against the same weights with fp16 activations the declaration prefills at 0.330x (Galaxy S26, 64 decoded tokens) and 0.496x (Pixel 8a) and decodes at 0.841x and 0.873x; no fp32 copy of the weights (Pixel 8a weight cache 1,342,110,512 vs 1,338,374,960 bytes) (card L107).
- Galaxy S26 GPU: the clock ceiling fell from 1,300 MHz during every 256-token run, so the GPU rates include throttled iterations; no unthrottled 256-token run was obtained. The GPU still took less time than the CPU in every run; on the Pixel 8a the CPU took 1.6 % less on average (card L36, L105).
- Decode logits padded to 49,156 columns: without it every GPU run printed 'TopK requires src tensor C dimension to be divisible by 4.' and, where the CPU began a reply with <|tool_call|> (id 49154), the GPU began with other text; with it the GPU begins those replies with <|tool_call|> (card L115).
- Tool calling: the bundle's model type (generic_model) has no tool-call parser, so a call arrives as reply text, <|tool_call|> followed by a JSON list; in the tests the model started a call only when the history already held one earlier round trip (card L72).
- Template: IBM's chat template with three additions for litert-lm 0.18 (string or part-list content, a tools block written by a macro that gives the bytes of tojson(indent=4), a tool role as IBM's tool_response turn); no start token, the stop token <|end_of_text|> (id 0) plus 14 strings from the template's turn ending; the bundle sets no sampler (card L70).
- Cache: without --cache, `run` writes an XNNPACK weight cache next to the model (1,346,661,944 bytes); without --backend, `run` uses the CPU (card L68).
- Accuracy: GSM8K questions 0-99 (greedy, at most 1024 output tokens, litert-lm 0.18.0 Python API): this file 35 / 35 / 35 on the Mac GPU, the same weights 39 / 35 on the Mac CPU, bf16 34; the model wrote the '#### <number>' line in none of the replies; 8-question check 8/8 on the Mac CPU and GPU, GPU 8/8 and CPU 7/8 on the Galaxy S26 and the Pixel 8a (card L40-L57).
- Not measured on iOS; no NPU row: the file is a dynamic-range int8 graph and was not tried on an NPU (card L109).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
