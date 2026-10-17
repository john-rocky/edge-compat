---
family: granite
license: apache-2.0
model_id: granite-4.0-micro-int8
source_url: https://huggingface.co/litert-community/granite-4.0-micro
task: text-generation
---

# granite-4.0-micro-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite |
| **Source** | https://huggingface.co/litert-community/granite-4.0-micro |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 + hf-to-litertlm tools/gpu_graph/gpu_graph_retrofit.py (the 2026-10-16 GPU-composite rewrite); repro = hf-to-litertlm granite_work/convert_granite40_micro.sh for the 2026-10-13 files (card L108-L109) litert-torch 0.9.4, litert-converter 0.4.0, ai-edge-quantizer 0.9.0, transformers 5.14.1, torch 2.13.0; packed by the litert-lm 0.18.0 CLI |
| **Command** | `ibm-granite/granite-4.0-micro rev 56111ae1, rewritten from GraniteMoeHybrid (40 attention layers, 0 experts) to the GraniteForCausalLM layout, weights unchanged -> export with the int8 recipe (6 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; the embedding table as its own section; no start token; IBM's template embedded) -> metadata edit (stop id 100257 only, max_num_tokens 4096) -> gpu_graph_retrofit.py --decomp exporter --mask add_bcast (odml.cache_update + odml.runtime_bmm composites and a param_tensor input; all weight tensors, metadata, tokenizer and embedder sections byte-identical) (card L109-L112)` |
| **Quantization** | int8 dynamic on linears, float activations; the embedding table in its own section; GPU-composite graph (card L24) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-micro_int8.litertlm` | `04ee6ad9e905448f02fa58ad3757cd3a32a773ce7b908fafb73cf8d9b5d84df5` | 3594.629 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-14/granite-4.0-micro-int8__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/granite-4.0-micro-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 192.685 | 21.355 | 5361.5 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1380.945 | 89.335 | 752.8 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-14 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 18.17 | 2.53 | 56810.0 | 4983.01 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-15 | measured |

## Pitfalls

- Re-ship: since 2026-10-16 the file carries the GPU-composite graph, the retrofit of the 2026-10-13 file with the same weights; against the 2026-10-13 file the Mac GPU takes 16 % less time for a 1024-token prefill plus 256 decoded tokens (card L18, L36, L109).
- Pixel 8a GPU: no row - the harness stops a process when MemAvailable falls below 400,000 kB and did so in every process of the 2026-10-13 int8 file's one try (307,820 / 381,800 / 106,656 kB); the re-shipped file has the same weights and 884,736 bytes more, and the try was not repeated (card L32, L104).
- Galaxy S26: the card's rows come from Google's Developer Device Platform (device m1q-36, one session per row, the litert CLI), not the own phone (card L18, L84).
- Tool calling: with a get_weather tool the file writes <tool_call>{...} </tool_call> as text, which the app has to parse (the bundle's generic model type has no tool-call parser); a tool result passed as a Python-API content part renders empty (the model then invented a weather report), passed as the tool message's text it was used (card L69, L71).
- Template: IBM's chat_template.jinja byte for byte with its default system message (replaced when you pass one); no start token, the only stop token <|end_of_text|> (id 100257), no thinking declared; the runtime renders nine of ten conversation shapes exactly as transformers does (card L69).
- Sampler: the bundle sets no sampler; `run` passes one only with --top-k, --top-p, --temperature or --seed (card L67).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (3.42 GB for this file); without --backend, `run` uses the CPU (card L67).
- Speed: int4 is the faster file on every GPU, this file on every CPU (Mac 17.30 vs 20.84 s, Pixel 8a 157.45 vs 167.05 s for the 1024 / 256 workload); not measured on iOS (the main weights section is over 2 GiB), not run on the NPU (card L29-L36, L104).
- Accuracy: GSM8K questions 0-99 (greedy, at most 1024 output tokens, litert-lm 0.18.0 Python API): this file 84 on the CPU and 90 on the Mac GPU, bf16 86; the same 90 questions right as the 2026-10-13 file on the GPU; 8-question check 8/8 on CPU and GPU, bf16 8/8; IFEval-lite not run (card L40, L46-L55).
- Pixel 8a CPU: the benchmark process started with the clocks uncapped, unlike the 2026-10-13 rows (card L104).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
