---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm2.5-1.2b-thinking-int8
source_url: https://huggingface.co/litert-community/LFM2.5-1.2B-Thinking
task: text-generation
---

# lfm2.5-1.2b-thinking-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-1.2B-Thinking |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.1 |
| **Command** | `python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Thinking out_lfm25_12b_fp --fp && python ../minicpm_work/quantize_litertlm.py apply out_lfm25_12b_fp/model.litertlm lfm25_int8.litertlm --recipe wi8fc` |
| **Quantization** | int8 dynamic, post-hoc linears + embedding (wi8fc); convs stay float |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-Thinking_int8.litertlm` | `91c044a117066c16e4e318ffc0ac0b2976b4a0af7fbd464054589c34e9220b28` | 1186.938 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/lfm2.5-1.2b-thinking-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-24/lfm2.5-1.2b-thinking-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm2.5-1.2b-thinking-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/lfm2.5-1.2b-thinking-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 357.09 | 35.55 | 600.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | load_failed | no | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 74.88 | 7.07 | 3561.7 | 1888.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The published file is the post-hoc linears-only recipe, per the HF card file table. The A/B in REPRODUCE.md records export-time conv-int8 as roughly neutral on this tune (-1 point), so the choice here is not forced the way it is on the JP tune — but the shipped bytes are the wi8fc build, and that is what this card describes.
- The convs are deliberately left float. Post-hoc ALL_SUPPORTED int8 through ai-edge-quantizer kills the conv layers of this hybrid outright (no output) — quantize_litertlm.py documents this on the wi8fc branch. Convs can only be quantized at export time, which is a different file, not this one.
- This artifact cannot use a GPU delegate. It is litert-torch 0.9.1 lineage, whose ShortConv patch emits GATHER_ND and INT64 ops that GPU delegates reject; the delegate takes 536 of 579 operations and engine creation then aborts. The count was measured on the Instruct int4 sibling of the same lineage (Pixel 8a and Galaxy S26, litert-lm v0.16.0); this file has not been separately gated. The repo ships no int8 GPU variant — the GPU re-export exists only for int4.
- The 0.9.1 exporter needs the ShortConv prefill-pad fix that convert_lfm25.py applies: the stock block saves its conv state from the padded columns of a prefill chunk, corrupting the first generated token of nearly every reply. It is easy to miss — the model recovers after about one token and GSM8K still parses answers, it just loses roughly 20 points.
- litert-lm >= 0.15 needs an ExecutorMetadata section for this hybrid: files exported before that run on 0.14 but fail at inference on 0.15 with 'missing some output TensorBuffers'. The published files were repaired in place on 2026-08-04.
- GSM8K (greedy, 0-shot CoT, n=100) is 77 for this int8 file against a bf16 reference of 81.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
