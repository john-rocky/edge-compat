---
family: granite
license: apache-2.0
model_id: granite-4.1-8b-int8
source_url: https://huggingface.co/litert-community/granite-4.1-8b
task: text-generation
---

# granite-4.1-8b-int8

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
| **Command** | `granite41_work/convert_granite41_8b.sh (hf-to-litertlm): export (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, no start token, IBM's chat_template.jinja embedded) -> metadata edit (stop id 100257 only, max_num_tokens 4096) -> tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast` |
| **Quantization** | int8 dynamic on linears, float activations; the embedding table as a separate section (card L24) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.1-8b_int8.litertlm` | `3ed1e7d4a3429e75a77b7445cbc4ce051aba888104eb254dcede457b5dfc70d9` | 8535.254 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/granite-4.1-8b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/granite-4.1-8b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 110.6 | 9.815 | 9361.4 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 717.96 | 46.07 | 1448.0 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (one CPU process) (card L18).
- GPU-composite graph: each layer's two DYNAMIC_UPDATE_SLICE KV writes are one odml.cache_update and the attention products odml.runtime_bmm; on the Mac GPU the rewrite took the 1024 / 256 workload from 8.20 to 6.95 s in the same measurement windows, and before the rewrite the same weights scored the same on GSM8K (95), no question changing between right and wrong (card L54, L98).
- Lightweight export: the default litert-torch export path reached 98.8 GB on a 128 GB Mac in the converter passes; with the 1 MiB default lazy-constant threshold the converter left the 40 o_proj weights unquantized behind a constant-only multiply by Granite's residual multiplier 0.22, so the export uses a 1e8-byte threshold (every FULLY_CONNECTED weight an int8 constant, no op with all-constant inputs). Export peaks 50-56 GiB (card L97).
- Chat template: IBM's chat_template.jinja byte for byte, no default system message (no system turn without one); no start token; the only stop token <|end_of_text|> (id 100257); no thinking declared. A tool result passed as a Python-API content part renders empty — pass it as the tool message's text (card L68).
- Tool calls come back as text: the model writes <tool_call>{"name": ..., "arguments": ...}</tool_call> and the app has to parse it (the bundle's generic model type has no tool-call parser) (card L70).
- Engine reuse: one Engine can serve one conversation after another (a second conversation gave the same replies as a fresh Engine in 3 of 3 runs, Mac GPU; KV cache only, no recurrent state) (card L70).
- GPU precision: the file runs the GPU with the runtime's default fp16 activations; fp32 activations declared gave the same eight check answers and ran 15 % slower, so none is declared (card L54).
- Galaxy S26: not run on this phone — after another 8B int8 file of the same recipe class dropped the phone off USB during its GPU engine warm-up (2026-10-14), the lane does not start 8B int8 engines on it; a rule, not a measured limit. No CPU row on either phone (batch-1 CPU peak factors put the file above MemTotal); Pixel 8a not tried (8.95 GB, more than the phone's RAM) (card L92).
- iOS not measured: the main weights section is 8,528,914,704 bytes, over 2 GiB (card L36).
- Accuracy: GSM8K n=100 (greedy, at most 1024 output tokens, fresh process per question, Mac GPU): bf16 96 / int8 95 — one question differs (right only in bf16) (card L46-L54).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
