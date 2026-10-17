---
family: lfm2.5
license: other (lfm1.0)
model_id: lfm2.5-350m-int4
source_url: https://huggingface.co/litert-community/LFM2.5-350M
task: text-generation
---

# lfm2.5-350m-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-350M |
| **License** | other (lfm1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 (released wheels); repro = hf-to-litertlm lfm_work/convert_lfm25_350m.sh (card L110) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, ai-edge-litert 2.2.0, transformers 5.14.1, torch 2.13.0 (released wheels); packed by the litert-lm 0.18.0 CLI |
| **Command** | `LiquidAI/LFM2.5-350M rev 9e6c6ccf: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; 22 state buffers: 10 ShortConv states, 6 key and 6 value caches) -> post-hoc int4 quantization -> a scan of the int4 block scales (none of the 92 blockwise tensors has a zero or subnormal scale) -> metadata repack with LiteRT-LM's lfm2 configuration (card L110-L111)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 lm_head + embedding (card L28) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-350M_int4.litertlm` | `272b0a1ca55f8b39fe636927deffe4049feb93b5eda7c05a5dbbb3141d6d53f8` | 236.271 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-12/lfm2.5-350m-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2.5-350m-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2.5-350m-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 875.875 | 90.63 | 1190.0 | 1120.91 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 4391.315 | 106.86 | 240.0 | 472.211 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 1195.41 | 123.11 | 864.9 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 13965.02 | 425.85 | 75.7 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 342.115 | 51.405 | 3015.0 | 835.961 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 1245.865 | 50.055 | 840.0 | 872.18 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; this file was not tried on litert-lm 0.17.1, where the int8 file of the same conversion answered as if no question had been asked (the runtime prefilled 9 tokens for a one-question turn where 0.18.0 prefilled 16) (card L21).
- Template: Liquid AI's chat template fails at template creation on litert-lm 0.18.0 ({% generation %} markers), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|>; marked non-thinking, with the configuration's thought channel kept (card L71, L112).
- One Engine per conversation: on the int8 file, an Engine that had already served one conversation changed 2 of the next conversation's 3 answers (Python API, CPU, greedy, 3 of 3 processes); a related open issue is LiteRT-LM #3165; the card advises a new Engine for each conversation (card L75).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0 (Liquid AI's recommended repetition_penalty 1.05 is not carried); `run` uses it unless --top-k, --top-p, --temperature or --seed is passed; --top-k 1 gives greedy output (card L69).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (231 MB for this file on CPU); without --backend, `run` uses the CPU (card L67).
- Tool calling is declared (model type lfm2); after a tool call the bundle template and Liquid AI's differ in two places (call arguments in double quotes, the tool result wrapped in [ … ]); on litert-lm 0.18.0 (Python API, CPU, greedy) the file returned a parsed get_weather call with {"city": "Paris"} (card L73).
- Which file is faster differs by device: int4 on the Galaxy S26 GPU (2.63 s vs int8 2.94 s for a 1024-token prefill plus 256 decoded tokens), int8 on the Pixel 8a GPU (5.77 vs 5.94 s: there int4 decodes at 50.06 vs 47.9 tok/s and prefills at 1245.9 vs 2410.6 tok/s), within 1 % on the Mac GPU (0.67 vs 0.68 s); 38.6 % smaller than the int8 file; not measured on iOS, not run on the NPU (card L30-L39, L101-L102, L106).
- Galaxy S26 CPU: the row ran with the clock lowering during it (lowered in 5 of 7 samples, read every 2 s); decode fell across the three iterations (101.67 -> 97.12 -> 84.14 tok/s) (card L97).
- Accuracy: IFEval-lite (first 120 rows of google/IFEval whose instruction types are checked mechanically, prompt-level strict, greedy; comparable only within that table) bf16 81 (67.5 %), this file 79 (65.8 %) on CPU and 78 (65.0 %) on the Mac GPU; 8-question check 7/8 on CPU and on GPU (missing "Which is larger: 0.9 or 0.11?" -> "0.11 is larger than 0.9."), bf16 7/8; in a three-turn CPU conversation it answered with a different word ("large") than bf16 and completed all three turns; GSM8K 8 of 100 (bf16 2) (card L43, L45, L49-L57).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L116).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
