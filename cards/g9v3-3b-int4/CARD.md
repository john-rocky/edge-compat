---
family: g9v3
license: apache-2.0
model_id: g9v3-3b-int4
source_url: https://huggingface.co/litert-community/G9v3-3B
task: text-generation
---

# g9v3-3b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | g9v3 |
| **Source** | https://huggingface.co/litert-community/G9v3-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 + gptqmodel 7.3.5 (calibration); repro = hf-to-litertlm g9v3_work/convert_g9v3_3b.sh (card L111) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1 (export); gptqmodel 7.3.5 on torch 2.14.1 (MPS) for the calibration; packed by the litert-lm 0.18.0 CLI |
| **Command** | `ai9stars/G9v3-3B rev d9553445: gptqmodel (4 bits, group 128, symmetric, 256 chat samples from ultrachat_200k, alpaca and the GSM8K training split at pinned revisions, thinking off) -> dequantized -> export (6 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; the embedding table as its own section) with the GPTQ scales recovered as blockwise-128 int4, 781 block scales below 2^-14 floored to it -> metadata edit (stops ids 1 and 130073, thought channel, thinking declared, max_num_tokens 4096) -> GPU activation type declared fp16 (card L111-L113)` |
| **Quantization** | GPTQ-calibrated int4 (group 128) carried as blockwise-128 int4 on linears, int8 lm_head + embedding; GPU activations declared fp16 (card L26) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `G9v3-3B_int4.litertlm` | `ea2c773ec678d44ffca43c0fca21a46622fd812fb9b53706f82d966d02e3c468` | 1749.455 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-13/g9v3-3b-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-13/g9v3-3b-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-13/g9v3-3b-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 402.855 | 18.465 | 2615.0 | 1182.48 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 | 2026-10-13 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1506.875 | 61.485 | 696.0 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-13 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 125.11 | 8.955 | 8295.0 | 1638.86 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-13 | measured |

## Pitfalls

- Calibrated int4: a data-free blockwise int4 export of this checkpoint scored 58 on GSM8K and its zero and subnormal block scales did not load on the CPU, so this file is GPTQ-calibrated; the calibration set is rebuilt by the script, not redistributed (alpaca's licence) (card L59, L112).
- GPU precision: this file declares fp16 GPU activations (GSM8K 81 with fp16, 78 with fp32) and runs fastest that way; the same weights declared fp32 took 5.29 s on the Mac GPU against 4.84 s and 25.09 s on the Galaxy S26 GPU against 16.41 s (card L37, L55-L56, L80, L98).
- Thinking on: in a three-turn conversation the third turn of this file returned no answer - its trace ran to the turn's 1536-token output limit without closing (the int8 file passed) (card L43).
- Galaxy S26: before each leg the phone rested until its clocks were at their top and the thermal status was 0 (up to 300 s); the clocks still lowered during every run (the GPU ceiling in 18 of 26 samples, the CPU clock in 45 of 47 for this file) (card L89, L98).
- Thinking: off by default and switched on per request; with thinking on the trace goes to the bundle's thought channel (<think> … </think>); a request that does not set the thinking switch is treated as thinking off, where the vendor template leaves it to the model (card L21, L74).
- Template: built on LiteRT-LM's models/qwen3 template, reading message content as parts; renders the same text as AI9Stars's template for 10 of 11 conversation shapes; stops ids 1 and 130073, start token <s> (card L74).
- Tool calling is not declared: with a get_weather tool the file writes the vendor's XML form as text, for the app to parse (card L76).
- Sampler: the bundle sets no sampler, so `run` is greedy unless you pass sampling flags; on the 0.18.0 CLI --temperature and --top-p take effect only together with --top-k (card L72).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (1.54 GB for this file); without --backend, `run` uses the CPU (card L72).
- Runtime: litert-lm 0.18.0 on the Mac; on litert-lm 0.17.1 this file answered a plain `run` with "The capital of France is Paris." (one CPU process) (card L19).
- Accuracy: GSM8K questions 0-99, thinking off, greedy, at most 1024 output tokens: this file 81 on the Mac GPU as shipped and 78 on the CPU, bf16 82; 8-question check 7/8 on CPU and GPU (the rhyme question); IFEval-lite not run (card L41, L47-L59).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
