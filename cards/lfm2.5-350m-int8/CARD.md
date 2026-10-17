---
family: lfm2.5
license: other (lfm1.0)
model_id: lfm2.5-350m-int8
source_url: https://huggingface.co/litert-community/LFM2.5-350M
task: text-generation
---

# lfm2.5-350m-int8

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
| **Command** | `LiquidAI/LFM2.5-350M rev 9e6c6ccf: unquantized export (11 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; 22 state buffers: 10 ShortConv states, 6 key and 6 value caches) -> post-hoc int8 quantization -> metadata repack with LiteRT-LM's lfm2 configuration (card L110-L111)` |
| **Quantization** | int8 dynamic on linears + int8 embedding, convs float (card L27) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-350M_int8.litertlm` | `6a78c7a2746386f8919476a34ef0ca40d596a5c44e979520fae8318da5cfb1f6` | 384.728 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-12/lfm2.5-350m-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-12/lfm2.5-350m-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2.5-350m-int8__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-12/lfm2.5-350m-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 1752.69 | 80.31 | 600.0 | 1274.03 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 5660.615 | 92.895 | 190.0 | 506.672 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | run_failed | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 2200.63 | 134.0 | 473.0 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 13556.05 | 423.73 | 78.0 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 509.06 | 41.81 | 2035.0 | 979.012 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 2410.57 | 47.895 | 445.0 | 939.234 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 | 2026-10-12 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 on a Mac; on litert-lm 0.17.1 this file answered as if no question had been asked - the runtime prefilled 9 tokens for a one-question turn where 0.18.0 prefilled 16, the full turn (card L21).
- Template: Liquid AI's chat template fails at template creation on litert-lm 0.18.0 ({% generation %} markers), so the bundle carries LiteRT-LM's lfm2 template (byte-identical to models/lfm2/chat_template.jinja at commit 86a7e6d0), which reads message content as a list of parts; stops <|im_end|> and <|endoftext|>, start token <|startoftext|>; marked non-thinking, with the configuration's thought channel kept (card L71, L112).
- One Engine per conversation: on an Engine that had already served one conversation, 2 of the next conversation's 3 answers changed - "What is 17 times 23?" got 323 where a fresh Engine answered 391 (this file, Python API, CPU, greedy, 3 of 3 processes); a related open issue is LiteRT-LM #3165 (card L75).
- Sampler: the bundle's default sampler is temperature 0.1, top-k 50, top-p 1.0 (Liquid AI's recommended repetition_penalty 1.05 is not carried); `run` uses it unless --top-k, --top-p, --temperature or --seed is passed; --top-k 1 gives greedy output (card L69).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (358 MB for this file on CPU); without --backend, `run` uses the CPU (card L67).
- Tool calling is declared (model type lfm2); after a tool call the bundle template and Liquid AI's differ in two places (call arguments in double quotes, the tool result wrapped in [ … ]); on litert-lm 0.18.0 (Python API, CPU, greedy) the file returned a parsed get_weather call with {"city": "Paris"} (card L73).
- Which file is faster differs by device: int8 on the Pixel 8a GPU (5.77 s vs int4 5.94 s for a 1024-token prefill plus 256 decoded tokens), int4 on the Galaxy S26 GPU (2.63 vs 2.94 s), within 1 % on the Mac GPU (0.68 vs 0.67 s); not measured on iOS, not run on the NPU (card L30-L39, L106).
- Galaxy S26 CPU: the row ran with the clock lowering during it (lowered in 3 of 6 samples, read every 2 s); decode fell across the three iterations (89.63 -> 87.28 -> 73.34 tok/s) (card L97).
- Accuracy: IFEval-lite (first 120 rows of google/IFEval whose instruction types are checked mechanically, prompt-level strict, greedy; comparable only within that table) bf16 81 (67.5 %), this file 82 (68.3 %) on CPU and 86 (71.7 %) on the Mac GPU; 8-question check 6/8 on CPU and on GPU (misses "What is 8 times 7?" -> "64" and "Which is larger: 0.9 or 0.11?"), bf16 7/8; GSM8K 2 of 100 (bf16 2) with most replies the final-answer line alone (card L43, L49-L57).
- License: the LFM Open License v1.0's section 5 conditions commercial use on an annual revenue below 10 million US dollars (card L116).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
