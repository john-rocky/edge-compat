---
family: spark-x2.5
license: apache-2.0
model_id: spark-x2.5-4b-int4
source_url: https://huggingface.co/litert-community/Spark-X2.5-4B
task: text-generation
---

# spark-x2.5-4b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | spark-x2.5 |
| **Source** | https://huggingface.co/litert-community/Spark-X2.5-4B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf path over the vendor modeling code patched for the registered attention interface (released wheels only; repro = hf-to-litertlm spark_work/patch_modeling.py + spark_work/convert_spark.py wrapping scripts/export_simple_template.py on the jinja path) 0.9.3 (transformers 5.14.1; ai-edge-quantizer named but unversioned in the sources — HF card Conversion notes; FINDINGS env ~/venvs/ltconv040dev; gates and GSM8K on litert-lm 0.17.0) |
| **Command** | `patch_modeling.py -> convert_spark.py <export_dir> <out> templates/spark25_think.jinja <recipe> (NO_START_TOKEN=1, USE_JINJA=1, CACHE 4096, PREFILL 1024,256,64,16,4,1 = 6 signatures — the granite-4.2 iPhone lesson; int4: EXTERNALIZE_EMBEDDER=1 BOCTAV4_128) -> scripts/set_activation_type.py --type fp32 on BOTH ship files (metadata-only repack, weights byte-identical) (FINDINGS bundles table + fp32 repack)` |
| **Quantization** | export-time int4 blockwise-128 + OCTAV on linears, int8 embedding externalized in its own section (EXTERNALIZE_EMBEDDER=1 BOCTAV4_128); prefer_activation_type fp32 declared in-bundle |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Spark-X2.5-4B_int4.litertlm` | `0b9ab47c1e65214995dc10f9833728941a73c05cfb219e3ddb05b0fe623b16a1` | 2367.207 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-09-07/spark-x2.5-4b-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-07/spark-x2.5-4b-int4__galaxy-s26.json`, `data/device_runs/0.17.0/2026-09-07/spark-x2.5-4b-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 213 | - | 34.77 | 5.47 | 6310.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| galaxy-s26 | gpu | pass | yes | - | 213 | - | 56.36 | 4.89 | 3980.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-09-07 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 117.84 | 18.37 | 2237.2 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 825.98 | 53.63 | 328.6 | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-07 | measured |

## Pitfalls

- int4 (block-128) is the recommended file: same accuracy as int8 (GSM8K 94 = bf16), 41% smaller (2.48 GB), and the faster GPU file on the Mac (53.6 vs 49.4 tok/s decode); unlike the 1.7B, block-128 shows no verbosity collapse here (mean thought 3,212 chars, 7 unfinished) — the block-32 hedge (2.67 GB, gates passed) was not needed and stays a lane record (HF card; FINDINGS 4B int4 b128).
- The iPhone int4 legs score 7/8 on both backends: the one miss is the rhyme line at the end of the composite 8-question prompt (it answers 'sweet'); asked alone the same question is answered 'blue' on every backend; the int4 Metal leg bottomed at ~2.0 GB free (HF card).
- GPU activations are declared fp32 in the bundle (prefer_activation_type = fp32, a metadata-only repack; weights byte-identical): the runtime runs the text decoder with fp16 activations on the GPU by default, and on this 36-layer decoder that silently costs reasoning accuracy — the same int8 weights scored GSM8K 84 under fp16 GPU activations and 94 under fp32, established by re-running the 11 GPU losses on the CPU fp32 path (10 came back right; the 5 controls stayed wrong). The 8-question gate did not see it (8/8 either way); only a long-generation accuracy row does. Expect a larger GPU memory footprint than an fp16-activation file of the same size (HF card Conversion notes; FINDINGS 4B GSM8K).
- Both files are at bf16 parity on GSM8K (94 = 94 = 94; n=100, greedy, 0-shot CoT, 3584 output tokens, own harness) once fp32 activations are declared; the 1.7B stays on the fp16 default because it measured exact parity there (HF card; FINDINGS).
- On the Galaxy S26 this 4B is a ~5 tok/s model on every measured path: the fp32-activation OpenCL path does not beat the CPU (int4: GPU 4.9 vs CPU 5.5-5.6 tok/s decode) but halves peak memory (2.0 vs 3.5 GB) — pick by memory, not speed; the int8 GPU --benchmark cell is unmeasured because a ~3,900-token thinking response at ~5 tok/s exceeded the 20-minute leg cap twice, although the file loads with full OpenCL delegation (2236/2236) and answers the gate on the GPU (HF card Performance; FINDINGS S26; DECISIONS #164(e)).
- iPhone 17 Pro (G41DeviceTest, --max-tokens 3072, Increased Memory Limit entitlement, iOS 27.0): the int8 file — ONE 4.24 GB weights section, no externalized embedder — loads and scores 8/8 on Metal and CPU (available-memory floor ~2.6 GB on the Metal leg, ~4.5 GB on CPU), so the ~2 GiB single-section mmap ceiling recorded in the Llama-3.2 era did not bite on this runtime lineage with this entitlement; the 4B int4 b128's 1.994 GiB main section was sized to it unnecessarily (harmless). Keep externalizing for int4 (the tied-vocab duplication reason stands), but do not cite the 2 GiB ceiling as a hard rule until re-measured on the runtime it came from (FINDINGS 'The iOS ~2 GiB single-section mmap ceiling did NOT bite').
- GPU init times on the iPhone are with a warm shader cache (a cold first launch of the int8 file initialised in 18.8 s); the int4 file's first Metal leg after a fresh 2.5 GB copy initialised in 13 s and then produced nothing for 25 min until the host timeout — one-off, cause not established, the re-run scored normally; the card carries the completed legs (HF card; FINDINGS).
- Reasoning model: give it a generous output budget (>= 2048 tokens, 3584 for math) — truncated mid-thought it produces no final answer at all; the bundle pre-fills the vendor think opener (<|Bot|><think>) and declares the thought channel (<think>…</think>) so the runtime separates reasoning from the answer and honours a thinking budget; without the channel the runtime streams raw reasoning into the answer and ignores any budget (HF card Usage + Conversion notes).
- The vendor modeling code is patched for export, not re-implemented: it computes attention through its own eager function and ignores config._attn_implementation, so the export copy dispatches through the registered attention interface (litert-torch's transposed KV cache), threads the per-call kwargs, declares the attention-backend flags and applies the per-head sigmoid output gate in the interface's layout; eager output is bit-identical to the vendor file (max |dlogit| 0.0 on 24 random tokens); two further edits make the vendor file load under transformers 5 at all (HF card Conversion notes; FINDINGS parity).
- No start token in the metadata: the tokenizer declares <|start_of_sentence|> as BOS but never prepends it (add_bos_token false); the template carries its own, so the exporter's unconditional start_token write was suppressed (measured harmless in bf16 on the 8Q gate, but it is not the vendor prompt) (HF card Conversion notes).
- Prompt format carried as a Jinja template verbatim: a default system block ('you are a helpful assistant.', a user system prompt appended after it), every message wrapped in <|start_of_sentence|> … <|end_of_sentence|>, stop <|end_of_sentence|>; tool-call formatting is not carried; KV budget 4096 (the original's 1M context does not apply on-device); the vendor's recommended sampling is temperature 1.0 / top-p 0.95 while every number on the card is greedy (HF card Usage).
- Tokenizer parity: the bundle's HF tokenizer.json section encodes 234/234 probe rows to the same ids as the tokenizers reading of the upstream file; multi-turn (3 turns, fact recall) passes with and without the runtime's channel filtering (HF card Correctness).
- iPhone 17 Pro rows come from the G41DeviceTest harness (composite 8-question prompt, --max-tokens 3072, byte count verified on-device): a generation gate, not a benchmark; the harness links swift-litert-lm's v0.15.0 pin and ran with the Increased Memory Limit entitlement — see the device rows' evidence (FINDINGS iPhone section; DECISIONS #164).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
