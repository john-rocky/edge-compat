---
family: olmo3
license: apache-2.0
model_id: olmo-3-7b-think-int8
source_url: https://huggingface.co/litert-community/Olmo-3-7B-Think
task: text-generation
---

# olmo-3-7b-think-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | olmo3 |
| **Source** | https://huggingface.co/litert-community/Olmo-3-7B-Think |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (stock export_hf path, lightweight conversion, embedding table as its own section); repro = hf-to-litertlm olmo3_work/ litert-torch 0.9.4 (litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0); packed and checked with litert-lm 0.18.0 |
| **Command** | `olmo3_work/convert_olmo3_7b_think.sh (hf-to-litertlm): allenai/Olmo-3-7B-Think at d97e442d -> export (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, the bundle template embedded) -> metadata edit (no start token, stop ids 100265 / 100257, the thought channel, max_num_tokens 4096) -> the repo's tokenizer.json as the tokenizer section` |
| **Quantization** | int8 dynamic on every linear layer (219 FULLY_CONNECTED per prefill signature, 225 in decode), int8 embedding table; activations float, the runtime's default fp16 on the GPU (card L31, L108) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Olmo-3-7B-Think_int8.litertlm` | `e075c5c92582fa1446723003225239b4dd131a07195ca49e4a7117388641bfbb` | 7090.256 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-16/olmo-3-7b-think-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-16/olmo-3-7b-think-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 131.36 | 6.695 | 7958.8 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 838.805 | 30.395 | 1253.7 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-16 | measured |

## Pitfalls

- Always thinks: the template ends the prompt with an opened <think>; the CLI's --thinking false and --thinking-budget 0 leave the output byte for byte the same, and a positive budget closes the thought channel at the budget while the model goes on reasoning in the visible text (with --thinking-budget 64 it wrote </think> as text, then the answer) (card L27, L79).
- Long reasoning inside a 4,096-token context: prompt, thought and answer share the KV budget; on GSM8K the mean reply was about 1,600 tokens and the 0.18.0 CLI has no output-length flag, so a long thought can reach the end of the context before the answer (card L27, L79).
- Accuracy: GSM8K n=100 (greedy, 2,048 output tokens, answers scored after </think>, a never-closed reply scored on its whole text, Mac GPU): bf16 84 / int8 81, unclosed thoughts 26 / 32 (card L57-L66).
- Chat template: Ai2's chat_template.jinja with three edits for the runtime's template engine (a content-parts macro, every content read through it, message.get(...) written as `is defined` tests); Ai2's default system message from the template file ("You are OLMo, a helpful function-calling AI assistant ..."), which differs from the README's "Default System Message" section; <think> / </think> are three plain tokens each (card L81).
- No start token; stop ids 100265 (<|im_end|>) and 100257 (<|endoftext|>); the bundle sets no sampler (every card number greedy; Ai2 recommends temperature 0.6 / top_p 0.95) (card L79, L81, L109).
- Tool calling is not wired: the runtime's tools= argument is not rendered (the template has no tools branch) and the generic model type has no tool-call parser; with Ai2's `functions` form this file (CPU) answered in text; a result in an `environment` turn got no answer inside 2,048 tokens (card L83).
- GPU sampler: every GPU run logs `TopK requires src tensor C dimension to be divisible by 4.` (vocabulary 100,278); the last two ids (<|endofprompt|>, <|pad|>) can never be sampled on the GPU (card L51).
- Precision: the same weights with fp32 GPU activations declared took 41 % more time (13.60 vs 9.64 s) and gave the same eight check answers, so the file keeps the runtime's fp16 default (card L96).
- Graph: the exporter's graph (no GPU composites): the GPU-composites rewrite that sped up two dense 8B files of the same converter refuses this graph (the attention-mask input also feeds the sliding-window mask); every layer keeps a full-length 4,096 cache, which equals the sliding window (card L25, L107).
- Galaxy S26: the int8 file was not started on the phone's GPU (a sibling 8B-class int8 file restarted the phone during its GPU engine warm-up) and the CPU was not run (peak arithmetic above the phone's memory); Pixel 8a not run; not run on an NPU (card L100-L102).
- iOS not measured: the decoder graph is one 6.53 GiB TFLite section, over the ~2 GiB single-section limit (card L45).
- Mac CPU: without --cache, `run` writes a compiled-weights cache next to the model on its first CPU run (6.90 GB for this file); the int8 file is the faster file on the Mac CPU (46.04 vs 51.97 s), the int4 file on the GPU (card L45, L79).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
