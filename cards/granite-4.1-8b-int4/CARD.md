---
family: granite
license: apache-2.0
model_id: granite-4.1-8b-int4
source_url: https://huggingface.co/litert-community/granite-4.1-8b
task: text-generation
---

# granite-4.1-8b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-4.1-8b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (lightweight export, 1e8-byte lazy-constant threshold) + hf-to-litertlm tools/gpu_graph/ GPU-composite rewrite; repro = hf-to-litertlm granite41_work/ litert-torch 0.9.4 (litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0); packed by the litert-lm 0.18.0 CLI |
| **Command** | `granite41_work/convert_granite41_8b.sh (hf-to-litertlm): export (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, no start token, IBM's chat_template.jinja embedded) with the int4 recipe -> int4 block-scale scan -> metadata edit (stop id 100257 only, max_num_tokens 4096) -> tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 embedding; the embedding table as a separate section (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.1-8b_int4.litertlm` | `7de1d77fb951572d12b8b17984d6418909b841e582d1df7866c5d08b3920ea60` | 4913.317 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/granite-4.1-8b-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/granite-4.1-8b-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-14/granite-4.1-8b-int4__pixel-8a.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/granite-4.1-8b-int4__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 90.9275 | 5.3475 | 11745.0 | 2087.335 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 51.5 | 11.68 | 19976.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 719.565 | 58.305 | 1440.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-14 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (one CPU process) (card L18).
- GPU-composite graph: each layer's two DYNAMIC_UPDATE_SLICE KV writes are one odml.cache_update and the attention products odml.runtime_bmm; on the Mac GPU the rewrite took the 1024 / 256 workload from 6.93 to 5.78 s in the same measurement windows, on the Galaxy S26 GPU the rewrite's decode gain is 1.4x, and before the rewrite the same weights scored the same on GSM8K (93), no question changing between right and wrong (card L54, L90, L98).
- Lightweight export: the default litert-torch export path reached 98.8 GB on a 128 GB Mac in the converter passes; with the 1 MiB default lazy-constant threshold the converter left the 40 o_proj weights unquantized behind a constant-only multiply by Granite's residual multiplier 0.22, so the export uses a 1e8-byte threshold (every FULLY_CONNECTED weight an int4 constant, no op with all-constant inputs); the int4 block-scale scan found none of 261,881,856 scales zero or subnormal. Export peaks 50-56 GiB (card L97-L98).
- Chat template: IBM's chat_template.jinja byte for byte, no default system message (no system turn without one); no start token; the only stop token <|end_of_text|> (id 100257); no thinking declared. Pass a tool result as the tool message's text: as a Python-API content part it renders empty (card L68).
- Tool calls come back as text: the model writes <tool_call>{"name": ..., "arguments": ...}</tool_call> and the app has to parse it (the bundle's generic model type has no tool-call parser) (card L70).
- Galaxy S26 GPU figures are warm-phone figures: both benchmark runs started at thermal status 3 (skin 45 C) and the GPU clock ceiling fell from 1300 to 342 MHz inside the runs (80 of 84 and 83 of 85 clock samples below 1300 MHz) (card L90).
- Pixel 8a: the one GPU try was stopped by the lane's memory rule (MemAvailable 327,604 kB, 55 s after the start, after prefill_1024 had been delegated 1860/1860 to LITERT_CL; the runtime prepares weights on the CPU side for Mali GPUs). No CPU row on either phone (batch-1 CPU peak factors put the file above both phones' RAM) (card L92).
- iOS not measured: the main weights section is 4,731,041,056 bytes, over 2 GiB (card L36).
- Accuracy: GSM8K n=100 (greedy, at most 1024 output tokens, fresh process per question, Mac GPU): bf16 96 / int4 93 — three questions differ, all right only in bf16 (card L46-L54).
- Mac backend choice: on the Mac CPU the int8 file is faster (its prefill is twice int4's); on both GPUs measured the int4 file is faster (card L36).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
