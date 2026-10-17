---
family: g9v3
license: apache-2.0
model_id: g9v3-3b-int8
source_url: https://huggingface.co/litert-community/G9v3-3B
task: text-generation
---

# g9v3-3b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | g9v3 |
| **Source** | https://huggingface.co/litert-community/G9v3-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4; repro = hf-to-litertlm g9v3_work/convert_g9v3_3b.sh (card L111) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (export); packed by the litert-lm 0.18.0 CLI |
| **Command** | `ai9stars/G9v3-3B rev d9553445: export with the int8 recipe (6 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; the embedding table as its own section; the bundle template embedded) -> metadata edit (stops ids 1 and 130073, thought channel <think> / </think>, thinking declared, max_num_tokens 4096) -> GPU activation type declared fp32 (card L111, L113)` |
| **Quantization** | int8 dynamic on linears, float activations; embedding table in its own section; GPU activations declared fp32 (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `G9v3-3B_int8.litertlm` | `c8c45d5bc3a5078290b38024651efda7b5a01cb1dce1d3b32284f9994d41afec` | 2958.768 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-13/g9v3-3b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-12/g9v3-3b-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int8__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 108.26 | 9.925 | 9580.0 | 4147.16 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 113.905 | 12.225 | 9085.0 | 1001.7 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-13 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1429.015 | 62.425 | 732.7 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-12 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 20.845 | 3.53 | 49440.0 | 3847.16 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |

## Pitfalls

- GPU precision: this file declares fp32 GPU activations; with the runtime's default fp16 it loses on the GPU (GSM8K 69 against 84 with fp32 and 85 on the CPU), so it runs slower on the GPU than the int4 file's fp16 form (card L37, L53-L54).
- Pixel 8a GPU: no row - the harness stops a process when MemAvailable falls below 400,000 kB and did so for this file with fp32 activations (lowest 300,700 kB in the first process, 0 in the benchmark) and with fp16 activations (393,560 kB) (card L32, L107).
- Pixel 8a CPU: the card's row is a run with no rest before it; a rested run of the same file was slower (152.59 s for the 1024 / 256 workload against 121.63 s), the CPU clock lowered in every sample either way (card L107).
- Galaxy S26: before each leg the phone rested until its clocks were at their top and the thermal status was 0 (up to 300 s); the clocks still lowered during every run (the GPU ceiling in 41 of 46 samples, the CPU clock in 45 of 46 for this file) (card L89, L98).
- Thinking: off by default and switched on per request; with thinking on the trace goes to the bundle's thought channel (<think> … </think>, printed by the CLI between [thought] and [/thought]); a request that does not set the thinking switch is treated as thinking off, where the vendor template leaves it to the model (card L21, L72, L74).
- Template: built on LiteRT-LM's models/qwen3 template, reading message content as parts; renders the same text as AI9Stars's template for 10 of 11 conversation shapes; stops ids 1 and 130073, start token <s> (card L74).
- Tool calling is not declared: with a get_weather tool the file writes the vendor's XML form as text (<function name="get_weather">…), for the app to parse (card L76).
- Sampler: the bundle sets no sampler, so `run` is greedy unless you pass sampling flags; on the 0.18.0 CLI --temperature and --top-p take effect only together with --top-k (AI9Stars recommends temperature 0.7 / top_p 0.95 with thinking off, 0.9 / 0.95 with thinking on) (card L72).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (2.73 GB for this file); without --backend, `run` uses the CPU (card L72).
- iOS not measured: the decoder weights section is 2,822,005,664 bytes, over 2 GiB (card L37).
- Runtime: litert-lm 0.18.0 on the Mac; on litert-lm 0.17.1 this file answered a plain `run` with "The capital of France is Paris." (one CPU process) (card L19).
- Accuracy: GSM8K questions 0-99, thinking off, greedy, at most 1024 output tokens (litert-lm 0.18.0 Python API): this file 85 on the CPU and 84 on the Mac GPU as shipped, bf16 82; 8-question check 8/8 on CPU and 7/8 on GPU (the rhyme question); no accuracy measured with thinking on; IFEval-lite not run (card L41, L47-L59).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
