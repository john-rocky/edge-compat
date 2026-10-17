---
family: granite
license: apache-2.0
model_id: granite-4.0-micro-int4
source_url: https://huggingface.co/litert-community/granite-4.0-micro
task: text-generation
---

# granite-4.0-micro-int4

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
| **Command** | `ibm-granite/granite-4.0-micro rev 56111ae1, rewritten from GraniteMoeHybrid (40 attention layers, 0 experts) to the GraniteForCausalLM layout, weights unchanged -> export with the int4 recipe (6 prefill signatures, 1024 down to 1, plus decode, over a 4096-position KV cache; the embedding table as its own section; no start token; IBM's template embedded) -> int4 block-scale scan (281 blockwise tensors, none zero or subnormal) -> metadata edit (stop id 100257 only, max_num_tokens 4096) -> gpu_graph_retrofit.py --decomp exporter --mask add_bcast (odml.cache_update + odml.runtime_bmm composites and a param_tensor input; all 281 weight tensors, metadata, tokenizer and embedder sections byte-identical; the main graph section grew by 884,736 bytes) (card L109-L112)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 embedding; the embedding table in its own section; GPU-composite graph (card L25) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-micro_int4.litertlm` | `b931889ac060fb563f5ae20b3f4ebc772297f15f77cdb9b0f55b1b25a160b325` | 2096.629 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-14/granite-4.0-micro-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/granite-4.0-micro-int4__galaxy-s26.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/granite-4.0-micro-int4__pixel-8a.json`, `data/device_runs/prebuilt-adac974b/2026-10-16/granite-4.0-micro-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 1024 | - | 235.5825 | 20.2425 | 4405.0 | 1437.76 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm prebuilt-adac974b · Android 16 (BP4A.251205.006) | 2026-10-15 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 102.42 | 23.615 | 10051.2 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 1391.1 | 111.25 | 745.1 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 | 2026-10-14 | measured |
| pixel-8a | cpu | fallback | no | - | 1024 | - | 12.2 | 3.085 | 84425.0 | 3658.71 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-15 | measured |
| pixel-8a | gpu | pass | yes | - | 1024 | - | 91.47 | 6.975 | 11340.0 | 2210.24 | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-16 | measured |

## Pitfalls

- Re-ship: since 2026-10-16 the file carries the GPU-composite graph, the retrofit of the 2026-10-13 file with the same weights; against that file the Mac GPU takes 17 % less time for the 1024 / 256 workload and the Galaxy S26 GPU 13 % less (two holds on the own phone, both orders) with the first token about 1.2 s later (prefill 335.6 -> 235.6 tok/s, decode 15.7 -> 20.2); on the Pixel 8a GPU about the same (two same-hold pairs within 2 % on their mean) (card L36, L93).
- Galaxy S26: the card's rows come from Google's Developer Device Platform (device m1q-36, one session per row, the litert CLI); the own Galaxy S26 (SM-S942Q) measured this file's GPU row within 3 % of them (two holds, both orders, the same binary) (card L84).
- The bool-mask form of the retrofit ran the Mac GPU 3 % and the Galaxy S26 GPU 10 % faster still but cost the CPU 8 % (Mac) to 9-23 % (Galaxy S26) and 1.2 GB of RSS, so it was not shipped (card L109).
- Tool calling: with a get_weather tool the file writes <tool_call>{...} </tool_call> as text, which the app has to parse (the bundle's generic model type has no tool-call parser); a tool result passed as a Python-API content part renders empty, passed as the tool message's text it was used (card L69, L71).
- Template: IBM's chat_template.jinja byte for byte with its default system message (replaced when you pass one); no start token, the only stop token <|end_of_text|> (id 100257), no thinking declared (card L69).
- Sampler: the bundle sets no sampler; `run` passes one only with --top-k, --top-p, --temperature or --seed (card L67).
- Cache: without --cache, `run` writes a compiled-weights cache next to the model (1.92 GB for this file); without --backend, `run` uses the CPU (card L67).
- Speed: this file is the faster one on every GPU (Mac 3.04 vs 3.61 s, Galaxy S26 16.70 vs 21.83 s, Pixel 8a 47.92 s with no int8 GPU row) and the slower on every CPU; 41.7 % smaller than the int8 file; not measured on iOS, not run on the NPU (card L27-L36, L104).
- Accuracy: GSM8K questions 0-99 (greedy, at most 1024 output tokens, litert-lm 0.18.0 Python API): this file 84 on the Mac GPU (the same 84 as the 2026-10-13 file; bf16 86); 8-question check 8/8 on CPU and GPU, bf16 8/8; IFEval-lite not run (card L40, L46-L55).
- Pixel 8a GPU: no gain from the composite graph on the Mali GPU; this row (a different day and hold) reads 15 % slower than the 2026-10-13 file's (card L104).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
