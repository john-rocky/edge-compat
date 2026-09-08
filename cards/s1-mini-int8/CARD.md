---
family: qwen3
license: other
model_id: s1-mini-int8
source_url: https://huggingface.co/mlboydaisuke/S1-mini-LiteRT
task: text-generation
---

# s1-mini-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/mlboydaisuke/S1-mini-LiteRT |
| **License** | other |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.3, stock export (hf-to-litertlm `s1mini_work/convert_s1mini.py`) 0.9.3 |
| **Command** | `export(model=<src with replaced chat template>, cache_length=4096, prefill_lengths=[1024,512,256,128,64,32,16,8,4,2,1]) -> strip the exporter's composite token_str stops` |
| **Quantization** | int8 dynamic on linears + embedding (exporter default recipe, dynamic_wi8_afp32) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `S1-mini_int8.litertlm` | `0376f042102cd1409055374d3e823faaaf3a66c5f61c6c744ffe01193d0ae397` | 656.278 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-25/s1-mini-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-25/s1-mini-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-25/s1-mini-int8__pixel-8a.json`, `data/device_runs/0.16.0/2026-09-05/s1-mini-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 254 | - | 175.34 | 12.34 | 1530.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 254 | - | 1027.54 | 25.8 | 290.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 280.73 | 15.53 | 739.0 | 1427.129 | iphone-17-pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-25 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 767.82 | 32.02 | 322.0 | 1697.05 | iphone-17-pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-25 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 426.65 | 33.32 | 630.0 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-25 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 3673.37 | 144.58 | 76.6 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-08-25 | measured |
| pixel-8a | cpu | fallback | no | - | 254 | - | 40.4 | 5.77 | 6460.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-25 | measured |
| pixel-8a | gpu | pass | yes | - | 254 | - | 434.01 | 12.28 | 670.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-25 | measured |

## Pitfalls

- Single-task model, not a chat model: it rewrites a question instead of answering it, so a general-knowledge question gate certifies nothing here. The gate that means something is exact-match against HF fp32 on the same rendered prompt (10/10 byte-for-byte, CPU and GPU, macOS).
- The upstream model requires enable_thinking=False, a flag no LiteRT-LM runtime passes. The bundle therefore ships a replaced chat template that hardcodes the non-thinking render and bakes in the exact system prompt the model was trained with; it was verified to render byte-identically to the vendor template under enable_thinking=False across all twelve upstream card examples, and a second turn's render string-extends the first turn plus its reply (the runtime's incremental-rendering requirement).
- litert-torch 0.9.3 emits punctuation-prefixed composite token_str stop entries ('.<|im_end|>\n', '?<|im_end|>\n', …), a SentencePiece-merge workaround. On a BPE tokenizer the merge never happens but the runtime still matches the multi-token stop and strips the WHOLE match, silently eating the reply's final './?/!'. Measured: 7 of 7 period-final test cases lost exactly that period, and a metadata-only rewrite keeping just the two real stop ids flipped all 7 back. Fatal on a punctuation-restoration model, and invisible to keyword-matching gates.
- Tied embedding/lm_head: an int4-on-linears + int8-on-embedding recipe describes one tensor two ways, and the quantizer resolves it by copying the vocabulary table per prefill signature — the int4 build came out LARGER than int8 (656 MB against 613 MB) until the embedder was externalized. int4 was then rejected on quality anyway (it drops commas and discourse words) and on speed (slower decode than int8 at 0.6B).
- Pixel 8a GPU is fully delegated (1301/1301 nodes, zero XNNPACK fallback) and 2x the CPU decode rate, but on number-dense input it deterministically drops a date separator ('March 3rd, 2026.' -> 'March 32026.'); the CPU backend on the same phone is byte-identical to macOS across all 12 cases. Backend choice here is exactness (CPU) against time-to-first-token (GPU).
- Google AI Edge Gallery builds current as of 2026-08 expose no accelerator choice for IMPORTED models and run them on the CPU — a property of the app's import path, not of the bundle, which delegates fully to the GPU on the same phone when the runtime drives it directly. Gallery's import dialog also defaults to sampling values (TopK 64, temperature 1.0); this model is greedy by design and needs TopK 1 / temperature 0.
- License carries a naming term: any use, distribution or integration must keep identifying the model as "S1-mini" by "Superwhisper", with that exact capitalization.
- All iPhone 17 Pro figures were recorded with the device reporting a 'serious' thermal state, so they are floors rather than peaks.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
