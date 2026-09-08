---
family: lfm2.5
license: other (lfm1.0)
model_id: lfm2.5-230m-int8
source_url: https://huggingface.co/litert-community/LFM2.5-230M
task: text-generation
---

# lfm2.5-230m-int8

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
| **Command** | `convert_230m.py OUTDIR --fp -> fix_230m_template.py raw.litertlm fixed.litertlm -> quantize_litertlm.py apply fp_fixed.litertlm X.litertlm --recipe wi8fc -> add_executor_metadata.py ... --litert-lm ~/venvs/lt0160run/bin/litert-lm (FINDINGS pipeline; mirror entry point hf-to-litertlm lfm_work/convert_lfm25_230m.py)` |
| **Quantization** | post-hoc dynamic int8 (wi8fc) on linears + embedding, convs float — applied to the --fp export, not the export-time conv-int8 recipe |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-230M_int8.litertlm` | `f8bc1a685e07e4f547d0390476b7d21b72c6dba3a7e4a76400b211b3d28628a9` | 253.728 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int8__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-09-01/lfm2.5-230m-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/lfm2.5-230m-int8__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 1006.37 | 94.69 | 210.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-01 | measured |
| galaxy-s26 | gpu | pass | yes | - | 205 | - | 3813.45 | 121.69 | 60.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-01 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | 1529.91 | 86.55 | 107.0 | 365.737 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 3422.14 | 143.19 | 75.0 | 472.94 | iPhone 17 Pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-09-01 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | 1727.0 | 153.82 | 154.8 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | 12809.74 | 561.24 | 21.8 | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-09-01 | measured |
| pixel-8a | cpu | fallback | no | - | 205 | - | 445.28 | 58.66 | 480.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| pixel-8a | gpu | pass | yes | - | 205 | - | 845.94 | 39.39 | 270.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 — the hybrid conv/KV state binds through the ExecutorMetadata section; tested on litert-lm 0.16.x (HF card header; manifest platform_notes).
- The upstream chat template does not run on the runtime's Jinja engine: {% generation %}/{% endgeneration %} (HF training-mask markers) is a minijinja parse error and message.get('content') a render error (map has no .get method), so a naive bundle dies on its first message. The embedded template strips the two markers and rewrites the two .get sites to plain indexing; rendering is byte-identical to the original across user/system/multi-turn/past-think/closed/tools conversations through HF apply_chat_template (HF card Conversion notes; FINDINGS make_metadata_230m.py / fix_230m_template.py).
- KV budget 4096, not the family's 4099: with the 1024-token prefill signature present a 4099 KV cache fails GPU engine creation at shader compile (CreateShaderModule validation error, macOS WebGPU); isolated to the pair prefill_1024 x cache 4099 — ladder+4096 and 128+4099 both compile, 4096 being a multiple of the widest signature. Shipped shape = 11 prefill signatures (1-1024) + 4096; on Galaxy S26 it delegates fully to OpenCL (492/492 nodes on every prefill signature, zero rejected ops) and both files pass Metal and CPU on iPhone 17 Pro (HF card Correctness + Conversion notes; FINDINGS GPU flip table).
- ExecutorMetadata and the second stop token are added post-export: the released 0.9.3 exporter omits ExecutorMetadata for this state-carrying architecture and litert-lm >= 0.15 needs it to bind the 8 conv states and 12 KV buffers; the exporter also derives only stop 7 (<|im_end|>) plus its punctuation string-stops, so stop id 2 (<|endoftext|>) is added to match the family [7, 2] scheme. Weights are byte-identical through both edits (HF card Conversion notes; FINDINGS).
- Convs stay float in this file: the obvious export-time int8 recipe (which also quantizes the conv layers) measured 53.3% on the same IFEval-style harness, 5 points below the shipped post-hoc wi8fc — the 230M lands on the JP/Thinking side of the family conv-int8 law, matching the LFM2.5-1.2B-JP and 2.6B conversions (HF card Accuracy; FINDINGS Quality A/B).
- Quality gates: IFEval-style strict (re-implementation of the mechanically checkable instruction types, n=120, prompt-level strict, greedy — comparable only within that table, not to published IFEval scores) bf16 60.0% / int8 58.3%, statistically indistinguishable on paired prompts; GSM8K n=100 bf16 23% / int8 23%, near floor for every configuration because the model is scoped away from math. 8-question sanity gate (Apple M4 Max, litert-lm 0.16.0): 7/8 CPU and 7/8 GPU vs bf16 8/8, the miss is a rhyme-completion line, no degeneration on any leg (HF card Correctness + Accuracy; manifest quantization row).
- Reproduction pins and traps: transformers 5.14.1 (5.15 breaks lfm2 export); quantize_litertlm.py shells out to litert-lm-builder and pyenv shims eat it (exit 127) unless the venv bin is first on PATH; the CLI's --no-template does not reproduce the internal-template stream (base-completion-style output, specials as literal text), so prove template fidelity at the render level, not through CLI A/B (FINDINGS).
- int8 is the recommended file everywhere except iPhone-Metal-decode-bound deployments: it is closer to bf16 on instruction following, decodes faster than int4 on Android GPU (Galaxy S26 Adreno) and is equal to int4 on desktop; int4 is the smaller file and the iPhone-Metal-decode option (HF card; manifest platform_notes).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
