---
family: mistral
license: apache-2.0
model_id: ministral3-8b-int4
source_url: https://huggingface.co/litert-community/Ministral-3-8B-Instruct-2512
task: text-generation
---

# ministral3-8b-int4

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
| **Command** | `ministral_work/convert_ministral3_8b.sh (hf-to-litertlm): scripts/extract_ministral3_text.py (Mistral3ForConditionalGeneration -> Ministral3ForCausalLM, the vision tower and the projector dropped) -> export (6 prefill signatures 1024..1 + decode, KV cache 4096, embedder externalized, the bundle template embedded) -> int4 block-scale scan -> metadata edit (stop id 2 only, max_num_tokens 4096) -> tools/gpu_graph/gpu_graph_retrofit.py --decomp exporter --mask add_bcast` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 embedding; the embedding table as a separate section (card L26); 239 blockwise tensors, 248,512,512 scales, none zero or subnormal (L100) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Ministral-3-8B-Instruct-2512_int4.litertlm` | `babb24c5d76bedcbc0b4418e2e6f07f8b92673df481fb4514848542daa50b160` | 4805.684 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/ministral3-8b-int4__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/ministral3-8b-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-14/ministral3-8b-int4__pixel-8a.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/ministral3-8b-int4__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 99.17 | 5.6425 | 10992.5 | 1981.73 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 55.39 | 12.93 | 18577.0 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 787.515 | 61.74 | 1316.6 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-14 | measured |

## Pitfalls

- Text only: the vision encoder of the release is not in these files; the weights come from the BF16 repo (the plain repo ships FP8 weights) (card L19, L97).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with "Paris" (one GPU process) (card L19).
- GPU-composite graph: each layer's two DYNAMIC_UPDATE_SLICE KV writes are one odml.cache_update and the attention products odml.runtime_bmm; on the Mac GPU the rewrite took the 1024 / 256 workload from 6.40 to 5.35 s in the same windows, on the Galaxy S26 its decode gain is 1.3× (the exported graph in the same two sessions: prefill 94.77 and 100.84 tok/s, decode 4.26 and 4.59 tok/s); before the rewrite the same weights scored 94 on GSM8K, two questions moved from wrong to right with it and none the other way (card L55, L91, L100).
- Lightweight export: the default litert-torch export path for an 8B model of this kind reached 98.8 GB on a 128 GB Mac before a RAM guard stopped it; with a 1e8-byte lazy-constant threshold every FULLY_CONNECTED weight is an int4 constant and no op has all-constant inputs. Export peaks 46-52 GB (card L99).
- Chat template: the Ministral-3-3B-Instruct-2512 conversion's template with the model name changed ([SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST], string and content-parts messages), Mistral AI's default system prompt verbatim ({today} / {yesterday} stay literal); the runtime adds the start token <s>; the only stop token </s> (id 2); no thinking declared. Known differences from the vendor render: no tool list, two text parts in one message concatenated, two user messages in a row become two turns (card L69).
- Tool calling is not wired: a tool does not change the prompt (533 tokens with or without one) and a `tool` message raises INVALID_ARGUMENT: No token IDs found; an app has to put tool definitions and results into the user text (card L71).
- Engine reuse: one Engine can serve one conversation after another (the second conversation gave the same replies as a fresh Engine in 3 of 3 runs, Mac GPU; KV cache only, no recurrent state) (card L71).
- Galaxy S26 GPU: measured warm — each benchmark started at thermal status 3 (skin 45 °C) with the GPU clock ceiling at 1300 MHz, and the ceiling fell to 282 MHz inside the runs; MemAvailable stayed above 864,440 kB. No CPU row on either phone (batch-1 CPU peak factors put the file above the S26's 11.66 GB and the Pixel 8a's 7.94 GB) (card L84-L93).
- Pixel 8a GPU: the one try was stopped by the lane's memory rule (a run ends at the first MemAvailable reading under 400,000 kB; here 316,676 kB, 56 s after the start, after prefill_1024 had been delegated 1586/1586 to LITERT_CL; the runtime prepares weights on the CPU side for Mali GPUs) (card L32, L93).
- iOS not measured: the main weights section is 4,488,362,336 bytes (4.18 GiB), over 2 GiB (card L37).
- Accuracy: GSM8K n=100 (greedy, at most 1024 output tokens, fresh process per question, Mac GPU): bf16 95 / int4 96 — three questions differ (one right only in bf16, two only in the int4 file); the int4 file is 41.7 % smaller than the int8 file (card L28, L47-L55).
- Mac CPU: without --cache, `run` writes a compiled-weights cache next to the model (4.49 GB for this file); the int4 file is the faster file on both GPUs measured, the int8 file on the Mac CPU (card L37, L67).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
