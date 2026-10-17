---
family: lfm2
license: other (lfm1.0)
model_id: lfm2-2.6b-int4
source_url: https://huggingface.co/litert-community/LFM2-2.6B
task: text-generation
---

# lfm2-2.6b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2 |
| **Source** | https://huggingface.co/litert-community/LFM2-2.6B |
| **License** | other (lfm1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4; repro = hf-to-litertlm lfm_work/convert_lfm2_26b.sh (card L110) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; packed by the litert-lm 0.18.0 CLI |
| **Command** | `LiquidAI/LFM2-2.6B rev 36ed799f: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache) -> post-hoc int4 quantization -> int4 zero-scale fix (746,432 zero block scales set to their tensor's smallest nonzero scale, block payload zeroed; every scale normal on re-scan) -> metadata repack with LiteRT-LM's lfm2 configuration (card L110-L111)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 lm_head + embedding (card L28) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2-2.6B_int4.litertlm` | `01c77fbd925f593014636573b024c162c6b910c4c1372cea010d54955e6942f8` | 1458.722 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-12/lfm2-2.6b-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2-2.6b-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 83.015 | 22.85 | 12375.0 | 3108.71 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 682.275 | 27.84 | 1540.0 | 1030.3 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 185.0 | 49.07 | 5555.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 2452.56 | 163.77 | 423.6 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 30.345 | 7.375 | 33935.0 | 2602.42 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 171.85 | 15.845 | 6020.0 | 1510.71 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; this file was not tried on litert-lm 0.17.1, where the int8 file of the same conversion answered as if no question had been asked (card L21).
- Block scales: the plain int4 quantization left 746,432 zero block scales; this file sets them to their tensor's smallest nonzero scale with the block payload zeroed, and every scale is normal on re-scan (card L111).
- Reasoning trace: a trace between <think> and </think> goes to the thought channel (the CLI prints it between [thought] and [/thought]); in the card's checks traces appeared for 1 of 120 IFEval-lite prompts per row and none for GSM8K questions, checks or phone prompts (card L71).
- Template: Liquid AI's chat template fails at render on litert-lm 0.18.0 (map has no method named get), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|> (card L71, L112).
- Tool calling is declared (model type lfm2): the file returned a parsed get_weather call (Python API, CPU, greedy) and used its result with automatic tool calling; with tools the template omits Liquid AI's tool-list and tool-response markers, wraps the tool result in [ … ] and quotes call arguments with double quotes (card L73).
- One Engine per conversation: the card advises a new Engine for each conversation (its probe on the int8 file, with two different system prompts, found the same factual answers on a shared Engine; LiteRT-LM issue #3165 reports recurrent state reused across conversations that share prompt tokens) (card L75).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0; Liquid AI recommends temperature 0.3, min_p 0.15 and repetition_penalty 1.05, which the bundle does not carry; --top-k 1 gives greedy output (card L69).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (1.51 GB for this file on CPU); without --backend, `run` uses the CPU (card L67).
- Speed: this file is the faster one on every GPU (Mac 1.98 vs 2.45 s, Galaxy S26 10.70 vs 13.47 s, Pixel 8a 22.11 vs 25.07 s for a 1024-token prefill plus 256 decoded tokens) and the slower on every CPU; 43.8 % smaller than the int8 file; not measured on iOS, not run on the NPU (card L30-L39, L106).
- Phone conditions: on the Galaxy S26 the GPU clock ceiling fell from 1300 to 826 MHz inside this file's GPU benchmark and the CPU clock was lowered in 34 of 35 samples of its CPU run; on the Pixel 8a the CPU clock was lowered in 63 of 66 samples of the CPU run (card L97, L106).
- Accuracy (greedy, at most 2048 output tokens, litert-lm on the Mac GPU, comparable only within the card's table): GSM8K 87 of 100 (bf16 88), IFEval-lite 87 of 120 (bf16 85); 8-question check 8/8 on CPU and 7/8 on GPU, bf16 7/8 (card L43, L49-L57).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L116).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
