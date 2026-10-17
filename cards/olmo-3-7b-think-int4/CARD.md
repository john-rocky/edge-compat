---
family: olmo3
license: apache-2.0
model_id: olmo-3-7b-think-int4
source_url: https://huggingface.co/litert-community/Olmo-3-7B-Think
task: text-generation
---

# olmo-3-7b-think-int4

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
| **Command** | `olmo3_work/convert_olmo3_7b_think.sh (hf-to-litertlm): allenai/Olmo-3-7B-Think at d97e442d -> export with the 2^-14 block-scale floor (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, the bundle template embedded) -> int4 block-scale scan -> metadata edit (no start token, stop ids 100265 / 100257, the thought channel, max_num_tokens 4096) -> the repo's tokenizer.json as the tokenizer section` |
| **Quantization** | int4 blockwise-32 (OCTAV) on every linear layer, block scales floored at 2^-14 (3,606,703 scales in 55 tensors; every scale normal after), int8 embedding table; activations float (card L32, L108) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Olmo-3-7B-Think_int4.litertlm` | `133ef829dcb98c1ad28f700cd3cd5f18e5fb60ae9bfb6f4efe52d60ef21969be` | 4109.819 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-16/olmo-3-7b-think-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-16/olmo-3-7b-think-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 64.275 | 7.105 | 16072.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 836.55 | 34.76 | 1252.9 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |

## Pitfalls

- Block scales: the plain int4 build wrote 1,574,919 zero and 2,031,784 subnormal fp16 block scales (of 215,210,752), and XNNPACK prepares a blockwise int4 layer only when every scale is a normal float, so this file floors every block scale at 2^-14 before the weights are rounded (card L108).
- Always thinks: the template ends the prompt with an opened <think>; the CLI's --thinking false and --thinking-budget 0 leave the output byte for byte the same, and a positive budget closes the thought channel at the budget while the model goes on reasoning in the visible text (card L27, L79).
- CPU reasons longer: on the 8-question check the rhyme question's thought was still open at the 2,048-token cap on the CPU (7/8) and closed at 2,125 tokens with a 4,096-token cap; the GPU closed it at 1,542 tokens; the cause was not measured (card L49).
- Accuracy: GSM8K n=100 (greedy, 2,048 output tokens, answers scored after </think>, a never-closed reply scored on its whole text, Mac GPU): bf16 84 / int4 80, unclosed thoughts 26 / 37 (card L57-L66).
- Tool calling is not wired: tools= is not rendered and the generic model type has no tool-call parser; with Ai2's `functions` form this file (GPU) announced the get_weather call and then wrote a made-up result (22 °C, sunny) as if the call had happened, with no call parsed (card L83).
- Chat template: Ai2's chat_template.jinja with three edits for the runtime's template engine; Ai2's default system message from the template file, which differs from the README's "Default System Message" section; no start token; stop ids 100265 / 100257; no sampler in the bundle (card L79, L81).
- GPU sampler: every GPU run logs `TopK requires src tensor C dimension to be divisible by 4.` (vocabulary 100,278); the last two ids can never be sampled on the GPU (card L51).
- Galaxy S26 (12 GB class): this 4,096-token-cache file did not build its GPU engine on a platform device-lab unit (all 7 signatures delegated, then free memory fell from 2.59 to 0.67 GB and the system server restarted); the 2,048-token-cache file of the same weights is the form that runs there (card L100).
- Pixel 8a not run: the file plus a 1-2 GiB KV cache is more than the phone's free memory at a session start; not run on an NPU (card L101-L102).
- iOS not measured: the decoder graph is one 3.62 GiB TFLite section, over the ~2 GiB single-section limit (card L45).
- Mac: the faster file on the GPU (8.59 vs 9.64 s, decode 34.8 vs 30.4 tok/s) and the slower on the CPU (blockwise prefill 64 vs 131 tok/s); without --cache, the first CPU `run` writes a 3.89 GB compiled-weights cache next to the model (card L45, L79).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
