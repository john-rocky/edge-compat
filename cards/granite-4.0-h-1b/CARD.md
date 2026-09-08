---
family: granite4h
license: apache-2.0
model_id: granite-4.0-h-1b
source_url: https://huggingface.co/ibm-granite/granite-4.0-h-1b
task: text-generation
---

# granite-4.0-h-1b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite4h |
| **Source** | https://huggingface.co/ibm-granite/granite-4.0-h-1b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + granite hybrid patch (hf-to-litertlm granite_work/convert_granite4h.py; patch adds export-cache layers for linear_attention, state-continuation tracing, prefill-pad guard) editable checkout 115a136 + granite_hybrid_litert_torch.patch (hf-to-litertlm granite_work/README.md) |
| **Command** | `git -C litert-torch-granite checkout 115a136 && git -C litert-torch-granite apply granite_hybrid_litert_torch.patch && PYTHONPATH=litert-torch-granite python convert_granite4h.py ibm-granite/granite-4.0-h-1b out_granite_1b` |
| **Quantization** | int8 dynamic on linears + embedding; convs and the selective scan stay float (ship commit ea409da; HF card recipe table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-h-1b_int8.litertlm` | `d04280eded8a133cc053df460e131b7102098295f97d550c10796d7bd98f8cbf` | 1553.615 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-12/granite-4.0-h-1b__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-24/granite-4.0-h-1b__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/granite-4.0-h-1b__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/granite-4.0-h-1b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 224 | - | 178.72 | 29.39 | 1290.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 225 | - | 232.64 | 24.06 | 1010.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| mac-studio-m4-max | cpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.16.0 | 2026-08-12 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 42.3 | 4.38 | 6280.1 | 2599.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Mamba2 conv/SSM state buffers bind through litert-lm >= 0.15's generalized state binding (ExecutorMetadata) — the 0.14 engine cannot bind them (HF card; 350m draft card wording).
- This 2026-08-04 artifact runs the selective scan as generic float ops on CPU only (stated plainly on the card of record); the GPU story belongs to the 2026-08-13 folded-scan re-ship, a different file (see header note).
- Correctness of record: FP parity corr 1.000000 at all 8 checked decode positions vs HF; 8Q sanity 8/8 == the HF reference; first-token sweep clean at prompt lengths 12-60 against the engine's real chunk plans (ship commit ea409da).
- Family BOS trap: the 350m sibling's greedy decoding flips with a BOS token (LlmMetadata start_token root cause, commit 5f3176d) — the 1b is BOS-robust, but verify start_token when reusing metadata across granite sizes.
- The 2026-08-12 compat_check loadability record on this card measured exactly these bytes (sha-verified against HF revision history); records dated after 2026-08-13 measure the re-shipped file instead.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
