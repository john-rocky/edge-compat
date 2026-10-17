---
family: mistral
license: apache-2.0
model_id: ministral3-8b-int8
source_url: https://huggingface.co/litert-community/Ministral-3-8B-Instruct-2512
task: text-generation
---

# ministral3-8b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | mistral |
| **Source** | https://huggingface.co/litert-community/Ministral-3-8B-Instruct-2512 |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (text tower extracted from the multimodal checkpoint; lightweight export, 1e8-byte lazy-constant threshold) + hf-to-litertlm tools/gpu_graph/ GPU-composite rewrite; repro = hf-to-litertlm ministral_work/ litert-torch 0.9.4 (litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0); packed by the litert-lm 0.18.0 CLI |
| **Command** | `ministral_work/convert_ministral3_8b.sh (hf-to-litertlm): scripts/extract_ministral3_text.py (Mistral3ForConditionalGeneration -> Ministral3ForCausalLM, the vision tower and the projector dropped) -> export (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, the bundle template embedded) -> metadata edit (stop id 2 only, max_num_tokens 4096) -> tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast` |
| **Quantization** | int8 dynamic on linears, float activations; the embedding table as a separate section (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ministral-3-8B-Instruct-2512_int8.litertlm` | `f2f6e6621fa84a7c5c295b7a190270aa77453417de707c94daf6c144e2f9418b` | 8238.622 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/ministral3-8b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ministral3-8b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 124.045 | 11.49 | 8343.3 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 784.83 | 46.66 | 1326.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Text only: the vision encoder of the release is not in these files; the weights come from the BF16 repo (the plain repo ships FP8 weights) (card L19, L97).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (one GPU process) (card L19).
- GPU-composite graph: each layer's two DYNAMIC_UPDATE_SLICE KV writes are one odml.cache_update and the attention products odml.runtime_bmm; on the Mac GPU the rewrite took the 1024 / 256 workload from 7.78 to 6.76 s in the same windows; before the rewrite the same weights scored 95 on GSM8K, two questions moved from wrong to right with it and none the other way (card L55, L100).
- Lightweight export: the default litert-torch export path for an 8B model of this kind reached 98.8 GB on a 128 GB Mac before a RAM guard stopped it; with a 1e8-byte lazy-constant threshold every FULLY_CONNECTED weight is an int8 constant and no op has all-constant inputs. Export peaks 46-52 GB (card L99).
- Chat template: the Ministral-3-3B-Instruct-2512 conversion's template with the model name changed ([SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST], string and content-parts messages), Mistral AI's default system prompt verbatim ({today} / {yesterday} stay literal); the runtime adds the start token <s>; the only stop token </s> (id 2); no thinking declared. Known differences from the vendor render: no tool list, two text parts in one message concatenated, two user messages in a row become two turns (card L69).
- Tool calling is not wired: a tool does not change the prompt (533 tokens with or without one) and a `tool` message raises INVALID_ARGUMENT: No token IDs found; an app has to put tool definitions and results into the user text (card L71).
- Engine reuse: one Engine can serve one conversation after another (the second conversation gave the same replies as a fresh Engine in 3 of 3 runs, Mac GPU; KV cache only, no recurrent state) (card L71).
- GPU precision: the file runs the GPU with the runtime's default fp16 activations; fp32 activations declared gave all eight check answers equal to bf16's (seven of eight with fp16) and ran 15 % slower, so none is declared (card L55).
- Galaxy S26: no row — in its one try (2026-10-14) the phone dropped off USB during the GPU engine warm-up, 4 min 15 s after the process started (prefill_1024 to prefill_4 delegated 1586/1586 to LITERT_CL), and came back restarted; the cause is not established and no 8B int8 engine ran on this phone again. No CPU row on either phone (batch-1 CPU peak factors put the file above MemTotal); Pixel 8a not tried (8.64 GB, more than the phone's RAM) (card L33, L93).
- iOS not measured: the main weights section is 8,088,047,040 bytes (7.53 GiB), over 2 GiB (card L37).
- Accuracy: GSM8K n=100 (greedy, at most 1024 output tokens, fresh process per question, Mac GPU): bf16 95 / int8 97 — four questions differ (one right only in bf16, three only in the int8 file) (card L47-L55).
- Mac CPU: without --cache, `run` writes a compiled-weights cache next to the model (7.97 GB for this file); on the Mac CPU this file is faster than the int4 file, its prefill being twice int4's (card L37, L67).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
