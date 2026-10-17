---
family: lfm2
license: other (lfm1.0)
model_id: lfm2-700m-int4
source_url: https://huggingface.co/litert-community/LFM2-700M
task: text-generation
---

# lfm2-700m-int4

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
| **Command** | `LiquidAI/LFM2-700M rev 86f49fc9: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache) -> post-hoc int4 quantization (no zero or subnormal block scale in its 92 blockwise tensors) -> metadata repack with LiteRT-LM's lfm2 configuration (card L110-L111)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 lm_head + embedding, convs float (card L28) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2-700M_int4.litertlm` | `4ecfc71299a304e97e8f0ef5134085ecc4b5d414755b8c1709f6b552d20250b9` | 458.521 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-12/lfm2-700m-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-700m-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-700m-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 346.84 | 57.12 | 3060.0 | 1652.52 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 2427.335 | 73.25 | 435.0 | 534.43 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 580.705 | 105.965 | 1782.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 7839.15 | 348.26 | 133.5 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 152.955 | 30.325 | 6780.0 | 1250.67 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 619.655 | 39.905 | 1675.0 | 1026.02 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; this file was not tried on litert-lm 0.17.1, where the int8 file of the same conversion answered as if no question had been asked (card L21).
- Template: Liquid AI's chat template fails at render on litert-lm 0.18.0 (message.get("content")), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|>; marked non-thinking, with the configuration's thought channel kept and no reply in the checks containing its strings (card L71, L112).
- Tool calling is declared (model type lfm2): the file returned a parsed get_weather call (Python API, CPU, greedy) and used its result with automatic tool calling; with tools the template omits Liquid AI's <|tool_list_start|> / <|tool_list_end|> and <|tool_response_start|> / <|tool_response_end|> markers, wraps the tool result in [ … ] and quotes call arguments with double quotes (card L73).
- One Engine per conversation: the card advises a new Engine for each conversation (its probe on the int8 file, two different system prompts, found no change on a shared Engine; LiteRT-LM issue #3165 reports recurrent state reused across conversations that share prompt tokens) (card L75).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0; Liquid AI recommends temperature 0.3, min_p 0.15 and repetition_penalty 1.05, which the bundle does not carry (no min_p or repetition penalty); --top-k 1 gives greedy output (card L69).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (465 MB for this file on CPU); without --backend, `run` uses the CPU (card L67).
- Speed: this file is faster than int8 on the Mac and Galaxy S26 GPUs (0.87 vs 0.97 s and 3.92 vs 4.64 s for a 1024-token prefill plus 256 decoded tokens) and equal on the Pixel 8a GPU (8.07 s each); on the CPUs it is faster on the Pixel 8a (15.13 vs 16.21 s) and slower on the Mac and the Galaxy S26; 40.5 % smaller than the int8 file; not measured on iOS, not run on the NPU (card L30-L39, L106).
- Galaxy S26 rows of this file: the GPU row ran with the GPU clock limit falling from 1300 to 1100 MHz in the last 2 of 7 samples (the unflagged first iteration read 2494.6 / 69.22 tok/s); the CPU row is from an earlier run that started on a hot phone (thermal status 3), the CPU clock lowered in 10 of 11 samples (card L93-L97).
- Pixel 8a: every run started at thermal status 1; the CPU clock was lowered in 13 of 17 samples of the CPU run and 5 of 11 of the GPU run (card L106).
- Accuracy (greedy, litert-lm on the Mac CPU, comparable only within the card's table): IFEval-lite 67 of 120 (bf16 77), GSM8K 52 and 57 of 100 with two wordings of the prompt (bf16 59 and 59); 8-question check 7/8 on CPU and GPU, the miss being the rhyme question as for bf16 (card L43, L49-L55).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L116).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
