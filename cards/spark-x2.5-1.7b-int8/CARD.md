---
family: spark-x2.5
license: apache-2.0
model_id: spark-x2.5-1.7b-int8
source_url: https://huggingface.co/litert-community/Spark-X2.5-1.7B
task: text-generation
---

# spark-x2.5-1.7b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | spark-x2.5 |
| **Source** | https://huggingface.co/litert-community/Spark-X2.5-1.7B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf path over the vendor modeling code patched for the registered attention interface (released wheels only; repro = hf-to-litertlm spark_work/patch_modeling.py + spark_work/convert_spark.py wrapping scripts/export_simple_template.py on the jinja path) 0.9.3 (transformers 5.14.1; ai-edge-quantizer named but unversioned in the sources — HF card Conversion notes; FINDINGS env ~/venvs/ltconv040dev; gates and GSM8K on litert-lm 0.17.0) |
| **Command** | `patch_modeling.py (transformers-5 load fixes + attention-interface dispatch, eager path bit-identical) -> convert_spark.py <export_dir> <out> templates/spark25_think.jinja <recipe> (NO_START_TOKEN=1, USE_JINJA=1, CACHE 4096, prefill ladder 1024..1 = 11 signatures; int4: EXTERNALIZE_EMBEDDER=1 BOCTAV4) (FINDINGS Conversion design + bundles table)` |
| **Quantization** | export-time dynamic int8 on linears + embedding (dynamic_wi8_afp32), one TFLite section (1.706 GiB); no post-processing |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Spark-X2.5-1.7B_int8.litertlm` | `96cd236da98fd95d1baa4ec391b166f359ded607cc13bc71cc9c0990556fe9bf` | 1749.616 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-09-07/spark-x2.5-1.7b-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-07/spark-x2.5-1.7b-int8__galaxy-s26.json`, `data/device_runs/0.17.0/2026-09-07/spark-x2.5-1.7b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 213 | - | 164.73 | 12.28 | 1370.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| galaxy-s26 | gpu | pass | yes | - | 213 | - | 636.84 | 15.82 | 400.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 820.35 | 41.0 | 336.5 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 2308.15 | 95.78 | 121.4 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |

## Pitfalls

- int8 is the recommended file: GSM8K (n=100, greedy, 0-shot CoT, 3584 output tokens, own harness, Mac GPU) 76 = bf16 76 (paired: both 71, bf16-only 5, int8-only 5; 20 unfinished at the cap on both) and 8/8 on every gate (HF card; FINDINGS GSM8K).
- The iPhone int8 legs score 7/8 on both backends: the one miss is the rhyme line at the END of the composite 8-question prompt (it answers 'purple'), the composite-format artifact already seen on granite-4.2; asked alone the same question is answered 'blue' on every backend (HF card Performance; FINDINGS iPhone).
- Reasoning model: give it a generous output budget (>= 2048 tokens, 3584 for math) — truncated mid-thought it produces no final answer at all; the bundle pre-fills the vendor think opener (<|Bot|><think>) and declares the thought channel (<think>…</think>) so the runtime separates reasoning from the answer and honours a thinking budget; without the channel the runtime streams raw reasoning into the answer and ignores any budget (HF card Usage + Conversion notes).
- The vendor modeling code is patched for export, not re-implemented: it computes attention through its own eager function and ignores config._attn_implementation, so the export copy dispatches through the registered attention interface (litert-torch's transposed KV cache), threads the per-call kwargs, declares the attention-backend flags and applies the per-head sigmoid output gate in the interface's layout; eager output is bit-identical to the vendor file (max |dlogit| 0.0 on 24 random tokens); two further edits make the vendor file load under transformers 5 at all (HF card Conversion notes; FINDINGS parity).
- No start token in the metadata: the tokenizer declares <|start_of_sentence|> as BOS but never prepends it (add_bos_token false); the template carries its own, so the exporter's unconditional start_token write was suppressed (measured harmless in bf16 on the 8Q gate, but it is not the vendor prompt) (HF card Conversion notes).
- Prompt format carried as a Jinja template verbatim: a default system block ('you are a helpful assistant.', a user system prompt appended after it), every message wrapped in <|start_of_sentence|> … <|end_of_sentence|>, stop <|end_of_sentence|>; tool-call formatting is not carried; KV budget 4096 (the original's 1M context does not apply on-device); the vendor's recommended sampling is temperature 1.0 / top-p 0.95 while every number on the card is greedy (HF card Usage).
- The int4 file externalizes the embedding table: the vocab is tied, and asking int4 for lm_head and int8 for the embedding makes the quantizer copy the 131,072-row table once per signature (4.6 vs 1.7 bytes/param on a tiny checkpoint), so the embedder lives in its own section; the int8 file needs no split (HF card Conversion notes; FINDINGS de-risk).
- Tokenizer parity: the bundle's HF tokenizer.json section encodes 234/234 probe rows to the same ids as the tokenizers reading of the upstream file; multi-turn (3 turns, fact recall) passes with and without the runtime's channel filtering (HF card Correctness).
- iPhone 17 Pro rows come from the G41DeviceTest harness (composite 8-question prompt, --max-tokens 3072, byte count verified on-device): a generation gate, not a benchmark; the harness links swift-litert-lm's v0.15.0 pin and ran with the Increased Memory Limit entitlement — see the device rows' evidence (FINDINGS iPhone section; DECISIONS #164).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
