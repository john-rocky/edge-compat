---
family: lfm2
license: other (lfm1.0)
model_id: lfm2-700m-int8
source_url: https://huggingface.co/litert-community/LFM2-700M
task: text-generation
---

# lfm2-700m-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2 |
| **Source** | https://huggingface.co/litert-community/LFM2-700M |
| **License** | other (lfm1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4; repro = hf-to-litertlm lfm_work/convert_lfm2_700m.sh (card L110) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; packed by the litert-lm 0.18.0 CLI |
| **Command** | `LiquidAI/LFM2-700M rev 86f49fc9: export with litert-torch's default int8 recipe dynamic_wi8_afp32 (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache) -> metadata repack with LiteRT-LM's lfm2 configuration (card L110-L111)` |
| **Quantization** | int8 weights on linears, the 10 convolutions and the embedding (litert-torch's default export recipe dynamic_wi8_afp32), float activations (card L27) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2-700M_int8.litertlm` | `ccc95007d868878ac749073cbc28b88097cb7d437fca3f9aec72f535cc7fd99b` | 770.837 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-12/lfm2-700m-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-12/lfm2-700m-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-700m-int8__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-700m-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 890.135 | 45.65 | 1200.0 | 1947.98 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 3396.395 | 58.965 | 320.0 | 517.246 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | run_failed | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 1134.25 | 90.87 | 913.9 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 7870.47 | 306.34 | 133.4 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 207.01 | 22.725 | 4990.0 | 1534.82 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 1451.05 | 34.75 | 735.0 | 1107.56 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; on litert-lm 0.17.1 the plain `run` of this file answered as if no question had been asked (card L21).
- Template: Liquid AI's chat template fails at render on litert-lm 0.18.0 (message.get("content")), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|>; marked non-thinking, with the configuration's thought channel kept and no reply in the checks containing its strings (card L71, L112).
- Tool calling is declared (model type lfm2): the file returned a parsed get_weather call (Python API, CPU, greedy) and used its result with automatic tool calling; with tools the template omits Liquid AI's <|tool_list_start|> / <|tool_list_end|> and <|tool_response_start|> / <|tool_response_end|> markers, wraps the tool result in [ … ] and quotes call arguments with double quotes (card L73).
- One Engine per conversation: in a probe with two different system prompts the shared Engine gave the same answers and token count as fresh Engines (3 of 3 processes); LiteRT-LM issue #3165 reports recurrent state reused across conversations that share prompt tokens, and the probe's two conversations started with different system prompts; the card advises a new Engine for each conversation (card L75).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0; Liquid AI recommends temperature 0.3, min_p 0.15 and repetition_penalty 1.05, which the bundle does not carry (no min_p or repetition penalty); --top-k 1 gives greedy output (card L69).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (747 MB for this file on CPU); without --backend, `run` uses the CPU (card L67).
- Greedy replies of this file differed between processes on the CPU path: the same file run twice on IFEval-lite flipped 5 prompts (24 prompts re-run gave byte-identical replies for 4) (card L57).
- Speed: int4 is faster on the Mac and Galaxy S26 GPUs, equal on the Pixel 8a GPU (8.07 s each for a 1024-token prefill plus 256 decoded tokens); on the CPUs this file is faster on the Mac (3.72 vs 4.18 s) and the Galaxy S26 (6.76 vs 7.43 s); not measured on iOS, not run on the NPU (card L30-L39, L106).
- Phone conditions: on the Galaxy S26 the harness waited for thermal status 0 and uncapped clocks before each process of this file's rows, and in its CPU run the CPU clock was lowered in 9 of 10 samples; on the Pixel 8a every run started at thermal status 1, and the CPU clock was lowered in 13 of 17 samples of the CPU run and 3 of 11 of the GPU run (card L97, L106).
- Accuracy (greedy, litert-lm on the Mac CPU, comparable only within the card's table): IFEval-lite 75 and 74 of 120 in two runs (bf16 77), GSM8K 58 and 62 of 100 with two wordings of the prompt (bf16 59 and 59); 8-question check 7/8 on CPU and GPU, the miss being the rhyme question as for bf16 (card L43, L49-L55).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L116).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
