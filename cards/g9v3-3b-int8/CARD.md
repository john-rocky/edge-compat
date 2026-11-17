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
| **Tool** | litert-torch 0.9.4, then hf-to-litertlm tools/gpu_graph/gpu_graph_retrofit.py; repro = hf-to-litertlm g9v3_work/convert_g9v3_3b.sh + the retrofit (card L113, L114) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (export); packed by the litert-lm 0.18.0 CLI; GPU-composite graph by tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast (2026-10-18) |
| **Command** | `ai9stars/G9v3-3B rev d9553445: export with the int8 recipe (6 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; the embedding table as its own section; the bundle template embedded) -> metadata edit (stops ids 1 and 130073, thought channel <think> / </think>, thinking declared, max_num_tokens 4096) -> GPU activation type declared fp32 -> the GPU-composite retrofit of that file without re-exporting (card L113, L114, L116)` |
| **Quantization** | int8 dynamic on linears, float activations; embedding table in its own section; GPU activations declared fp32; GPU-composite graph (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `G9v3-3B_int8.litertlm` | `8d9f5e8522e553458ab3fb7f4a461359367934d0741c5e82cd777da55ce69dc5` | 2959.877 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-18/g9v3-3b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-17/g9v3-3b-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int8__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int8__pixel-8a.json`, `data/device_runs/prebuilt-adac974b/2026-10-18/g9v3-3b-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 1024 | - | 108.26 | 9.925 | 9580.0 | 4147.16 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 113.905 | 12.225 | 9085.0 | 1001.7 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-18 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 263.8 | 24.86 | 3927.2 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-17 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1340.1 | 84.17 | 776.8 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-17 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 34.48 | 4.68 | 29910.0 | 3766.36 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-18 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |

## Pitfalls

- Graph form: since 2026-10-18 this file carries the GPU-composite graph, rebuilt from the 2026-10-13 int8 file without re-exporting (tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast): the two KV-cache writes per layer become one odml.cache_update composite, each attention matmul an odml.runtime_bmm composite, and a param_tensor input is added to every signature; all 470 weight buffers, the metadata, tokenizer and embedding sections and the fp32 activation declaration are byte-identical to the 2026-10-13 file; the decoder section grew by 1,163,264 bytes (card L114).
- Composite effect: against the 2026-10-13 file the composite graph shortens the GPU time by 21 % on the Mac and by 20 % on the platform's Galaxy S26, and lengthens the CPU time on that Galaxy S26 by 4-10 % (two session pairs, one in each order); on the Mac CPU the two files are within 4 % (card L37, L98).
- GPU precision: this file declares fp32 GPU activations; with the runtime's default fp16 it loses on the GPU (GSM8K 69 against 85 on the CPU and 84 on the Mac GPU with fp32), so it declares fp32 and runs slower on the GPU (card L37, L52-L54).
- Galaxy S26: this file's rows ran on the platform's device m1q-36 on Google's Developer Device Platform, one session per row through the litert CLI (a warm-up process, then a measured process of three prefill-and-decode cycles; rates and TTFT are the CLI's summary, the median of cycles 2 and 3); the farm's phone is allocated per session, so these rows are not same-hold pairs with the int4 file's own-phone rows (card L89).
- Pixel 8a GPU: no row - the harness stops a process when MemAvailable falls below 400,000 kB and did so for the 2026-10-13 int8 file with fp32 activations (lowest 300,700 kB in the first process, 0 in the benchmark) and with fp16 activations (393,560 kB); this file has the same weights and 1,163,264 bytes more, and the try was not repeated (card L32, L109).
- Pixel 8a CPU: the row is one run of a 2026-10-18 hold that ran both files (this file first; the CPU clock lowered in 67 of 80 samples); with the same rest before each run this phone's CPU results move by up to 9 % between holds, so the 2026-10-13 int8 file was not paired with this one on this phone and its earlier card row (121.63 s for the workload) is not comparable (card L100, L109).
- Thinking: off by default and switched on per request; with thinking on the trace goes to the bundle's thought channel (<think> … </think>, printed by the CLI between [thought] and [/thought]); a request that does not set the thinking switch is treated as thinking off, where the vendor template leaves it to the model (card L21, L72, L74).
- Template: built on LiteRT-LM's models/qwen3 template, reading message content as parts; renders the same text as AI9Stars's template for 10 of 11 conversation shapes; stops ids 1 and 130073, start token <s> (card L74).
- Conversation preface with tools and no messages: LiteRT-LM renders the chat template with an empty message list, and the template reads messages[0] without checking that the list is non-empty, so the render fails with 'Failed to apply template: undefined value' when the preface is prefilled at creation (prefillPrefaceOnInit, default false; creating the conversation fails) and in the preface render (renderPrefaceIntoString and its C++ and C API forms); the same with extra context (such as enable_thinking) and no messages, and in the preface render with an empty preface (run in the runtime on one Qwen3 template; the other templates rendered with minijinja 2.14 and jinja2); read in the source of LiteRT-LM 0.17.0, 0.18.1 and main, measured with the 0.18.1 PyPI package through its C API on these files' template text; the default path, where the first message is rendered with the preface, is not affected (the 0.18.1 Python API and CLI expose neither the flag nor the preface render); a system message or any initial message in the conversation config avoids it (card L120).
- Tool calling is not declared: with a get_weather tool the file writes the vendor's XML form as text (<function name="get_weather">…), for the app to parse; the template, tool-call and engine-reuse tests ran on the 2026-10-13 file, whose metadata sections (template included), tokenizer section and embedding section this file carries byte for byte (card L76).
- Sampler: the bundle sets no sampler, so `run` is greedy unless you pass sampling flags; on the 0.18.0 CLI --temperature and --top-p take effect only together with --top-k (AI9Stars recommends temperature 0.7 / top_p 0.95 with thinking off, 0.9 / 0.95 with thinking on) (card L72).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (2.73 GB for this file); without --backend, `run` uses the CPU (card L72).
- iOS not measured: the decoder weights section is 2,823,168,928 bytes, over 2 GiB (card L37).
- Runtime: litert-lm 0.18.0 on the Mac; on litert-lm 0.17.1 this file answered a plain `run` with "The capital of France is Paris." (one CPU process, re-run on 2026-10-18) (card L19).
- Accuracy: GSM8K questions 0-99, thinking off, greedy, at most 1024 output tokens (litert-lm 0.18.0 Python API): this file 84 on the Mac GPU as shipped, the same 84 questions as the 2026-10-13 file (0 discordant); the CPU row (85) is the 2026-10-13 file's (the CPU path runs the same arithmetic); bf16 82; 8-question check 8/8 on CPU and 7/8 on GPU (the rhyme question), the 2026-10-13 file's answers on both backends; no accuracy measured with thinking on; IFEval-lite not run (card L41, L47-L59).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
