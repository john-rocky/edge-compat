---
family: qwen3.8
license: apache-2.0
model_id: qwen38-27b-mixed-int4
source_url: https://huggingface.co/litert-community/Qwen3.8-27B
task: text-generation
---

# qwen38-27b-mixed-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3.8 |
| **Source** | https://huggingface.co/litert-community/Qwen3.8-27B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (commit 115a1360 + the Qwen3.5 hybrid patch of hf-to-litertlm; lightweight export, constants of 1e8 bytes and more stay lazy) + post-hoc ai-edge-quantizer; repro = hf-to-litertlm qwen35_work/convert_qwen38_27b.sh litert-torch 115a1360 + patch, transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0; packed and checked with litert-lm 0.18.1 |
| **Command** | `convert_qwen38_27b.sh (hf-to-litertlm; Qwen/Qwen3.8-27B at 1d4bf0f2): text-decoder export through litert-torch's lightweight conversion from an exact F32 copy of the checkpoint read through a memory map (6 prefill signatures 1024 / 256 / 64 / 16 / 4 / 1 + decode, a 4096-position KV budget, 128 state buffers) -> int4 blockwise-128 / int8 quantization with the 96 in_proj_a / in_proj_b projections float -> the thinking-off template, the repo's tokenizer.json, stop ids 248044 / 248046, fp32 GPU activations and max_num_tokens 4096; the script run from the download to the checks on Qwen/Qwen3.5-2B (2026-10-22 / 10-23) finished with exit 0 and gave this file's metadata section byte for byte` |
| **Quantization** | Mixed INT4: int4 blockwise-128 min-max on the 400 linear layers of the 64 blocks, int8 channelwise on the lm_head and the embedding, the 96 decay/gate projections of the 48 linear-attention layers float; 694 of the 190,054,400 block scales raised to a floor of 2^-14, every scale a normal float (none zero or subnormal); fp32 GPU activations declared (card L30, L92) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.8-27B_mixed_int4.litertlm` | `7396292ecec0741f166fb57ed65198c34a3f854b40e01951310e0b94f79d449f` | 14666.987 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.1/2026-10-23/qwen38-27b-mixed-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | sig:one prompt (CLI run --backend cpu --cache disk; the card's usage command 2) | - | - | - | - | 4156.039 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-23 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:8-question check (CLI run --backend gpu --cache no, one process per question) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-23 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:one prompt (CLI run --backend gpu --cache no; the card's usage command 1) | - | - | - | - | 24494.097 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-23 | measured |
| mac-studio-m4-max | gpu | pass | - | - | sig:three-turn conversation, thinking off (Python API) | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-23 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 176.99 | 18.42 | 5840.6 | 24896.16 | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.1 · macOS 27.0 (26A428) | 2026-10-23 | measured |

## Pitfalls

- Text only: the text decoder of the multimodal release (26,895,998,464 of its 27,781,427,952 parameters); no image or video input and no multi-token prediction. Only the 16 attention layers keep a growing KV cache (a 4,096-token budget); the 48 linear-attention layers carry a constant-size conv and recurrent state (card L24, L94).
- Thinking off only: the template writes an empty think block at every generation prompt and does not read enable_thinking or reasoning_effort; no thought channel, supports_thinking false. With a thinking switch in the same file (thinking on, thinking_token_budget 300, Mac GPU) the third turn of a three-turn conversation came back empty in 3 of 3 runs at litert-lm 0.18.1's default settings; with filter_channel_content_from_kv_cache off it answered (one run) (card L26).
- Desktop only: one 15.38 GB file, measured on a Mac only; it does not fit in the memory of the phones tested (MemTotal 11,389,756 kB on a Galaxy S26, 7,754,700 kB on a Pixel 8a); not run on iOS, and LiteRT has no NPU accelerator on macOS (card L20, L85, L86).
- The GPU engine takes 3½ to 4 minutes to build on the Mac M4 Max (the first usage command took 217 s from start to answer; a benchmark process spends 214–247 s per engine); the CLI prints about twice that as its init time (LiteRT-LM issue 4057). Build it once and keep it for the conversation (card L67).
- The CPU needs the weight cache: without it (--cache no) the CPU engine packs the int4 weights once for each of the 7 signatures, and on the block-32 file of the same export the benchmark's memory footprint grew from 5.8 GB to 91 GB in 48 s and the run was stopped. With --cache disk the first CPU run writes a 14.0 GB XNNPACK weight cache next to the model (120 s at a 4.2 GB peak footprint); with the cache in place the benchmark's peak was 4.1 GB and its engine build 122–125 s (card L68).
- fp32 GPU activations are declared: with the runtime's fp16 default the same weights answered the first of the 8 check questions with a run of `!` to the token cap (one run, 2026-10-22), while running the benchmark workload in 17.32 s against 19.87 s (card L32, L38, L83).
- One Engine per conversation (LiteRT-LM issue 3165, open): a new Conversation on an Engine that ran another one can continue from its recurrent state; the probe on this file (litert-lm 0.18.1, CPU) did not show the effect in 3 shared-Engine runs, so create a new Engine for each conversation all the same (card L69).
- One Engine per process on the Mac GPU: across 11 benchmark processes (4 engines each) on this file and variants of it, WebGPU validation errors appeared in 2 of the 44 engines, each the third engine of its process, on the block-32 file and on a copy without the fp32 declaration; none in this file's 12 benchmark engines or its 115 single-engine processes (card L70).
- Sampling: without --backend, run uses the bundle's default backend, the CPU; the bundle sets no sampler and the card's checks ran greedy. To sample as Qwen recommends for the non-thinking mode, pass --top-k 20 --top-p 0.8 --temperature 0.7 together (card L71).
- Tool calling: the template renders a tool list and tool responses in Qwen's format, but supports_function_calling is false and litert-lm 0.18.1's generic_model type has no parser for Qwen's XML tool call: a call comes back as text (the regex tool-call configuration of LiteRT-LM main, 2026-10-20, the fix for issue 3927, is not in 0.18.1 and this file carries none); not tried on this file (card L72).
- Tokenizer: the tokenizer section is the repo's tokenizer.json byte for byte (the exporter's own form tokenized Hindi, Thai, Tamil and Arabic with harakat differently); the runtime's tokenizer matches the tokenizers library on it in 292 of 292 strings (card L93).
- Which bytes the card's numbers ran on: unless a line names the published file, the checks and benchmarks ran on an earlier build with a thinking switch that differs only in the LlmMetadata section (TFLite, tokenizer and executor sections byte for byte the same; the published template renders the same thinking-off prompts in 216 of 216 renders). On the published file: the 8 questions 8/8 and the three-turn conversation with thinking off on the Mac GPU, and the three usage commands as printed (`42`, `42`, prefill 176.99 / decode 18.42 tok/s) (card L36, L65).
- Accuracy: GSM8K questions 0–99 (thinking off, greedy, at most 2,048 output tokens, Mac GPU, on the earlier file): bf16 96 / Mixed INT4 97; question 12 is counted right only in this file and its reply reached the 2,048-token cap without a final line; the two give the same answer on 98 of 100 (card L32, L48-L55).
- Mac speed (litert-lm 0.18.1, 1024-token prefill + 256 decoded tokens at context 2048, the mean of two benchmark processes on the earlier file): GPU prefill 175.50 / decode 18.24 tok/s at a 24.9 GB peak; CPU with the weight cache 21.82 / 5.59 tok/s at 4.1 GB; the block-32 file of the same export was slower on both (21.31 s / 103.34 s against 19.87 s / 92.74 s for the workload) (card L32, L80-L83).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
