---
family: zamba2
license: apache-2.0
model_id: zamba2-1.2b-instruct-int8
source_url: https://huggingface.co/litert-community/Zamba2-1.2B-instruct
task: text-generation
---

# zamba2-1.2b-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | zamba2 |
| **Source** | https://huggingface.co/litert-community/Zamba2-1.2B-instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch (reproduction script + patch: hf-to-litertlm zamba2_work/) (README Conversion notes) TODO (the README names the converter but not its version) |
| **Command** | `TODO (hf-to-litertlm zamba2_work/; invocation not stated in the README)` |
| **Quantization** | post-hoc dynamic int8 over linears + embedding only; convs and the scan stay float; fp32 activations declared for GPU (README file table + Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Zamba2-1.2B-instruct_int8.litertlm` | `af51316bd21f96c3766e81deba1ccc935ec0b1e7dedcfee2d23acc7686e49b0d` | 1364.136 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/zamba2-1.2b-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/zamba2-1.2b-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/zamba2-1.2b-instruct-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 213 | - | 87.79 | 12.21 | 2510.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 214 | - | 115.54 | 11.68 | 1940.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 24.26 | 1.81 | 11103.2 | 2993.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- iPhone 17 Pro (Metal): the 8-item composite probe answers 8/8 on CPU; on GPU 6/8, where both misses (8x7 and the rhyme) are questions the HF fp32 reference itself answers incorrectly on this composite — the GPU path tracks the reference model's own behavior (README Correctness).
- Galaxy S26 GPU per the README: runs, 54510/54510 ops across 12 subgraphs on LiteRT GPU (3620 MB); this repo's S26 CPU row is from the S7 backfill, the Pi 5 row from wave 2 (device runs).
- Requires litert-lm >= 0.15 (README intro).
- Correctness (README): float export matches the HF model teacher-forced across 48 decode positions (top-1/top-5 identical, corr 1.0000, KL ≈ 0); 8-question gate 8/8 on every lane (GPU and CPU, litert-lm 0.15.0 and 0.16.0), no degeneration, no greedy flips; hermetic prefill-chunk sweep clean (CPU fills 12-51, GPU 12-31).
- Hybrid packaging: folded selective scan (batched matmuls, all tensors rank <= 4 — what makes the graph fully delegable on GPU), min-only dt clamp handling (padded prefill positions forced to exact identity steps AFTER the clamp), shared block + adapters traced once per position with tied weights stored once, composite hybrid cache layer, prefill-pad guard (README Conversion notes).
- Streaming detokenization: the tokenizer's Strip decoder is removed from the bundle — Zamba2's metaspace (SP-BPE) tokenizer otherwise loses every interior space under the runtime's per-token streaming decode; the only behavior change is a sequence-initial space, which the runtime trims (README Conversion notes).
- License Apache-2.0, inherited from the Zyphra base (README front matter).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
