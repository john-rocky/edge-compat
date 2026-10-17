---
family: olmo3
license: apache-2.0
model_id: olmo-3-7b-think-int4-c2048
source_url: https://huggingface.co/litert-community/Olmo-3-7B-Think
task: text-generation
---

# olmo-3-7b-think-int4-c2048

| | |
|---|---|
| **Task** | text-generation |
| **Family** | olmo3 |
| **Source** | https://huggingface.co/litert-community/Olmo-3-7B-Think |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (stock export_hf path, lightweight conversion, embedding table as its own section) with the int4 block scales floored at 2^-14 before rounding; repro = hf-to-litertlm olmo3_work/ litert-torch 0.9.4 (litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0); packed and checked with litert-lm 0.18.0 |
| **Command** | `olmo3_work/convert_olmo3_7b_think.sh (hf-to-litertlm): the int4 export with a 2,048-position KV budget (6 prefill signatures 1024..1 + decode, embedder externalized, the bundle template embedded, the 2^-14 block-scale floor) -> int4 block-scale scan -> metadata edit (no start token, stop ids 100265 / 100257, the thought channel, max_num_tokens 2048) -> the repo's tokenizer.json as the tokenizer section` |
| **Quantization** | int4 blockwise-32 (OCTAV) on every linear layer, block scales floored at 2^-14, int8 embedding table; activations float; the same weights and recipe as Olmo-3-7B-Think_int4.litertlm (card L33, L108) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Olmo-3-7B-Think_int4_c2048.litertlm` | `47faae994b95e1ff7ab3b28a86cefcb8cec90cd96f987c7f865c7ace2f5eb0cc` | 4109.803 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-16/olmo-3-7b-think-int4-c2048__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 942.14 | 46.575 | 1108.4 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |

## Pitfalls

- Context of up to 2,048 tokens: prompt, thought and answer share a 2,048-token KV budget; with a 1,900-token output budget GSM8K scored 77 (47 unclosed thoughts; 12 replies used the whole budget and 37 more ended at the 2,048-token context), against 80 for the 4,096-token-cache int4 file; the two files (the same weights) gave the same final number on 71 of 100 (card L64, L66).
- Galaxy S26 (12 GB class): the form that builds its GPU engine there - on a platform device-lab unit it ran the 1024 / 256 workload in 41.3 s (prefill 211.8, decode 7.03 tok/s; peak 2.33 GB RSS; all 7 decoder signatures on the OpenCL delegate) on a unit that started clock-capped, so a platform-session figure, not a headline for a phone in hand; the 4,096-token-cache files did not build there (card L40, L100).
- Mac GPU: 23 % less time than the int4 file (6.58 vs 8.59 s; decode 46.6 vs 34.8 tok/s) at a context of 2,048 tokens (card L45, L96).
- Always thinks: --thinking false and --thinking-budget 0 change nothing; a positive budget closes the thought channel while the model goes on reasoning in the visible text (card L27, L79).
- Tool calling is not wired and the chat template, stop ids and GPU-sampler note are those of the int4 file: tools= not rendered, no tool-call parser, no start token, stop ids 100265 / 100257, the last two ids never sampled on the GPU (card L51, L81, L83).
- Pixel 8a not run and GPU execution on a Mali phone not verified; not run on an NPU; iOS not measured (decoder section over ~2 GiB) (card L45, L101-L102).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
