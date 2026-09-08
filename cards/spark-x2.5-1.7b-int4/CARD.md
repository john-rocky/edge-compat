---
family: spark-x2.5
license: apache-2.0
model_id: spark-x2.5-1.7b-int4
source_url: https://huggingface.co/litert-community/Spark-X2.5-1.7B
task: text-generation
---

# spark-x2.5-1.7b-int4

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
| **Quantization** | export-time int4 blockwise-32 + OCTAV on linears, int8 embedding externalized in its own section (EXTERNALIZE_EMBEDDER=1 BOCTAV4); no post-processing |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Spark-X2.5-1.7B_int4.litertlm` | `72075d8d46c74457c7883d0c3e3a60a5bfe406d324fd1155ffbcb04dd81d24ea` | 1204.613 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-09-07/spark-x2.5-1.7b-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-07/spark-x2.5-1.7b-int4__galaxy-s26.json`, `data/device_runs/0.17.0/2026-09-07/spark-x2.5-1.7b-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 213 | - | 82.78 | 13.6 | 2650.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| galaxy-s26 | gpu | pass | yes | - | 213 | - | 458.65 | 17.77 | 520.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 271.02 | 38.24 | 971.0 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 2492.1 | 103.62 | 112.4 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |

## Pitfalls

- int4 (block-32) is the phone / size option at 1.26 GB: it costs 10 GSM8K points (66 vs bf16 76; paired both 61, bf16-only 15, int4-only 5) — 9 of the 15 losses hit the 3584-token cap mid-think (int4 thinks longer) and 6 finished wrong; re-running the 15 GPU losses on the CPU fp32 path recovers only 2, so the loss is the int4 weights (block-32 OCTAV), not the GPU activation dtype (HF card; FINDINGS 1.7B GSM8K).
- Block-128 int4 was built and dropped: on the same first 61 questions block-32 scores 41 (17 cap hits) and block-128 21 (39 cap hits) — b128 reasons about twice as long (7,240 vs 3,846 mean thought chars) and meets the budget, the SmolLM2-1.7B 'block128 truly degrades at 1.7B' pattern (FINDINGS).
- int4 decodes faster than int8 on the Mac GPU (103.6 vs 95.8 tok/s) and on the Galaxy S26 (GPU 17.4-17.8 vs 13.9-15.8, CPU 13.6-13.9 vs 12.0-12.3) and scored 8/8 on the iPhone 17 Pro on both backends; on the S26 the GPU is the path for either file (same-or-faster decode, 2x the CPU prefill, less than half the CPU path's peak memory) (HF card).
- Reasoning model: give it a generous output budget (>= 2048 tokens, 3584 for math) — truncated mid-thought it produces no final answer at all; the bundle pre-fills the vendor think opener (<|Bot|><think>) and declares the thought channel (<think>…</think>) so the runtime separates reasoning from the answer and honours a thinking budget; without the channel the runtime streams raw reasoning into the answer and ignores any budget (HF card Usage + Conversion notes).
- The vendor modeling code is patched for export, not re-implemented: it computes attention through its own eager function and ignores config._attn_implementation, so the export copy dispatches through the registered attention interface (litert-torch's transposed KV cache), threads the per-call kwargs, declares the attention-backend flags and applies the per-head sigmoid output gate in the interface's layout; eager output is bit-identical to the vendor file (max |dlogit| 0.0 on 24 random tokens); two further edits make the vendor file load under transformers 5 at all (HF card Conversion notes; FINDINGS parity).
- No start token in the metadata: the tokenizer declares <|start_of_sentence|> as BOS but never prepends it (add_bos_token false); the template carries its own, so the exporter's unconditional start_token write was suppressed (measured harmless in bf16 on the 8Q gate, but it is not the vendor prompt) (HF card Conversion notes).
- Prompt format carried as a Jinja template verbatim: a default system block ('you are a helpful assistant.', a user system prompt appended after it), every message wrapped in <|start_of_sentence|> … <|end_of_sentence|>, stop <|end_of_sentence|>; tool-call formatting is not carried; KV budget 4096 (the original's 1M context does not apply on-device); the vendor's recommended sampling is temperature 1.0 / top-p 0.95 while every number on the card is greedy (HF card Usage).
- The int4 file externalizes the embedding table: the vocab is tied, and asking int4 for lm_head and int8 for the embedding makes the quantizer copy the 131,072-row table once per signature (4.6 vs 1.7 bytes/param on a tiny checkpoint), so the embedder lives in its own section; the int8 file needs no split (HF card Conversion notes; FINDINGS de-risk).
- Tokenizer parity: the bundle's HF tokenizer.json section encodes 234/234 probe rows to the same ids as the tokenizers reading of the upstream file; multi-turn (3 turns, fact recall) passes with and without the runtime's channel filtering (HF card Correctness).
- iPhone 17 Pro rows come from the G41DeviceTest harness (composite 8-question prompt, --max-tokens 3072, byte count verified on-device): a generation gate, not a benchmark; the harness links swift-litert-lm's v0.15.0 pin and ran with the Increased Memory Limit entitlement — see the device rows' evidence (FINDINGS iPhone section; DECISIONS #164).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
