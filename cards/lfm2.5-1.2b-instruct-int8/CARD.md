---
family: lfm2.5
license: lfm-open-license-v1.0
model_id: lfm2.5-1.2b-instruct-int8
source_url: https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct
task: text-generation
---

# lfm2.5-1.2b-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct |
| **License** | lfm-open-license-v1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.1 |
| **Command** | `python convert_lfm25.py LiquidAI/LFM2.5-1.2B-Instruct out_lfm25_12b` |
| **Quantization** | int8 dynamic, export-time recipe (linears + convs + embedding) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-1.2B-Instruct_int8.litertlm` | `e002a91545cec6328be2a86e9b8b4fccb2dd9dde7e6c8b29d730f915a0a697fe` | 1189.319 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-09-05/lfm2.5-1.2b-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/lfm2.5-1.2b-instruct-int8__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-01/lfm2.5-1.2b-instruct-int8__raspberry-pi-5.json`, `data/device_runs/0.17.0/2026-09-06/lfm2.5-1.2b-instruct-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 205 | - | 491.11 | 40.96 | 440.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.17.0 | 2026-09-06 | measured |
| pixel-8a | cpu | fallback | no | - | 205 | - | 143.38 | 20.4 | 1480.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 74.95 | 7.02 | 3558.2 | 1895.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Export-time int8 is what makes this file work: it is the only path that safely quantizes the conv layers. Post-hoc ALL_SUPPORTED int8 through ai-edge-quantizer kills them outright (no output), so post-hoc recipes must stay on linears and the embedding (wi8fc, wi4b32_wi8) — REPRODUCE.md, LFM2.5 family.
- Conv-int8 sensitivity is per-finetune, not a family property: export-time conv-int8 is free on this Instruct tune (+2 GSM8K) but costs the JP tune 9 points, which is why the published JP int8 file uses a linears-only recipe instead. A/B the two before reusing this command on a new finetune.
- This artifact cannot use a GPU delegate. It is litert-torch 0.9.1 lineage, whose ShortConv patch emits GATHER_ND and INT64 ops that GPU delegates reject; the delegate takes 536 of 579 operations and engine creation then aborts. That count was measured on the int4 sibling (Pixel 8a and Galaxy S26, litert-lm v0.16.0) — the same export lineage, but this specific file has not been separately gated on GPU. The repo ships no int8 GPU variant; the GPU re-export exists only for int4.
- The 0.9.1 exporter needs the ShortConv prefill-pad fix that convert_lfm25.py applies: the stock block saves its conv state from the padded columns of a prefill chunk, corrupting the first generated token of nearly every reply. Without it this int8 file scores 59 on GSM8K instead of 81 — that is the bug, not the quantization.
- litert-lm >= 0.15 needs an ExecutorMetadata section for this hybrid: files exported before that run on 0.14 but fail at inference on 0.15 with 'missing some output TensorBuffers'. The published files were repaired in place on 2026-08-04.
- Decode speed depends strongly on the KV budget: the HF card measures int8 decode at 101 tok/s with --max-num-tokens 1024 and 77 tok/s at 4096 on an M4 Max. Set the smallest budget the use case needs.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
