---
family: lfm2.5
license: other (lfm1.0)
model_id: lfm2.5-230m-int4
source_url: https://huggingface.co/litert-community/LFM2.5-230M
task: text-generation
---

# lfm2.5-230m-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-230M |
| **License** | other (lfm1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (released wheels only; repro = hf-to-litertlm lfm_work/convert_lfm25_230m.py + fix_230m_template.py) 0.9.3 (transformers 5.14.1 pinned — 5.15 breaks lfm2 export; ai-edge-quantizer named but unversioned in the sources; ExecutorMetadata retrofit via litert-lm 0.16.0 — FINDINGS.md ~/venvs/ltconv040dev + ~/venvs/lt0160run) |
| **Command** | `convert_230m.py OUTDIR --fp -> fix_230m_template.py raw.litertlm fixed.litertlm -> quantize_litertlm.py apply fp_fixed.litertlm Y.litertlm --recipe wi4b32_wi8 --algo octav -> fix_zero_block_scales.py (measured 0 patches, family no-op) -> add_executor_metadata.py ... --litert-lm ~/venvs/lt0160run/bin/litert-lm (FINDINGS pipeline; mirror entry point hf-to-litertlm lfm_work/convert_lfm25_230m.py)` |
| **Quantization** | post-hoc OCTAV int4 blockwise-32 on linears + int8 embedding (recipe wi4b32_wi8 --algo octav) over the --fp export; zero-block-scale fix measured as a no-op |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-230M_int4.litertlm` | `a4683cdafbf7b0ae526ba688fa905fd3db8546fc58221d10082443d4d0a6315c` | 168.568 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int4__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 262.9 | 50.68 | 800.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-01 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 1101.08 | 41.62 | 210.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-01 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 808.46 | 93.9 | 206.0 | 368.143 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 3036.13 | 161.7 | 71.0 | 625.205 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 1612.88 | 149.35 | 165.5 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 12862.27 | 555.27 | 21.7 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 — the hybrid conv/KV state binds through the ExecutorMetadata section; tested on litert-lm 0.16.x. Same template and shape notes as the int8 file (HF card header; manifest platform_notes).
- The upstream chat template does not run on the runtime's Jinja engine: {% generation %}/{% endgeneration %} is a minijinja parse error and message.get('content') a render error (map has no .get method), so a naive bundle dies on its first message. The embedded template strips the two markers and rewrites the two .get sites to plain indexing; rendering is byte-identical to the original across six conversation shapes through HF apply_chat_template (HF card Conversion notes; FINDINGS make_metadata_230m.py / fix_230m_template.py).
- KV budget 4096, not the family's 4099: with the 1024-token prefill signature present a 4099 KV cache fails GPU engine creation at shader compile (CreateShaderModule validation error, macOS WebGPU); isolated to the pair prefill_1024 x cache 4099. Shipped shape = 11 prefill signatures (1-1024) + 4096; on Galaxy S26 it delegates fully to OpenCL (492/492 nodes on every prefill signature, zero rejected ops) and passes Metal and CPU on iPhone 17 Pro (HF card Correctness + Conversion notes; FINDINGS GPU flip table).
- ExecutorMetadata and the second stop token are added post-export: the released 0.9.3 exporter omits ExecutorMetadata for this state-carrying architecture and litert-lm >= 0.15 needs it to bind the 8 conv states and 12 KV buffers; the exporter derives only stop 7 (<|im_end|>) plus its punctuation string-stops, so stop id 2 (<|endoftext|>) is added to match the family [7, 2] scheme. Weights are byte-identical through both edits (HF card Conversion notes; FINDINGS).
- Quality cost of int4 at this size: IFEval-style strict (n=120, prompt-level strict, greedy; comparable only within that table) 53.3% vs bf16 60.0% and int8 58.3% — about 7 points of instruction following, the price of the 89 MB saving; GSM8K n=100 22% vs bf16 23% (near floor, rules out collapse only). 8-question sanity gate (Apple M4 Max, litert-lm 0.16.0): 6/8 CPU (misses the rhyme-completion line plus one translation question) and 8/8 GPU, no degeneration on any leg; the cost is visible in the iPhone yardstick sample ('There are 5 days in a week'), which int8 answers correctly (HF card Accuracy + Correctness; FINDINGS ship-artifact rows; manifest quantization row).
- Not the fast file everywhere: at 230M the int4-vs-int8 speed ordering is per-device — int4 out-decodes int8 only on iPhone 17 Pro Metal; on Galaxy S26 Adreno OpenCL int4 decodes markedly slower than int8 on both GPU and CPU (the blockwise dequant overhead dominates at this size — the opposite of the larger LFM2.5 files), and on Mac the two are a wash. Pick int4 for size or iPhone-only deployments, not for Android speed; int8 is the recommended file (HF card; manifest platform_notes; FINDINGS device gates).
- Measurement caveat: the int4 GPU prefill on Galaxy S26 shows cold shader-compile variance between run 1 and run 2 — quote the pair, never one number (FINDINGS S26 table; manifest evidence note).
- Reproduction pins and traps: transformers 5.14.1 (5.15 breaks lfm2 export); quantize_litertlm.py shells out to litert-lm-builder and pyenv shims eat it (exit 127) unless the venv bin is first on PATH; the CLI's --no-template does not reproduce the internal-template stream, so prove template fidelity at the render level, not through CLI A/B (FINDINGS).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
