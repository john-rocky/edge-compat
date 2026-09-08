---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1-1.5b-deep-instruct-int8
source_url: https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct
task: text-generation
---

# falcon-h1-1.5b-deep-instruct-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/litert-community/Falcon-H1-1.5B-Deep-Instruct |
| **License** | other (falcon-llm-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch (reproduction script + patch: hf-to-litertlm falcon_h1_work/) (README Conversion notes) TODO (the README names the converter but not its version) |
| **Command** | `TODO (hf-to-litertlm falcon_h1_work/; invocation not stated in the README)` |
| **Quantization** | post-hoc dynamic int8 over linears + embedding only; convs and the scan stay float; fp32 activations declared for GPU (README file table + Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1-1.5B-Deep-Instruct_int8.litertlm` | `759b3b361ccd9b6138767fe1771f35d3be64301308b9a178d3edd43819447526` | 1748.166 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/falcon-h1-1.5b-deep-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/falcon-h1-1.5b-deep-instruct-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/falcon-h1-1.5b-deep-instruct-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 212 | - | 103.14 | 11.97 | 2140.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 212 | - | 120.58 | 6.0 | 1920.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 24.59 | 2.05 | 10897.7 | 3367.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (README intro).
- Correctness (README): float export matches the HF model teacher-forced across 48 decode positions (top-1/top-5 identical, corr 1.0000, KL ≈ 0); 8-question gate 8/8 on every lane (GPU and CPU, litert-lm 0.15.0 and 0.16.0); hermetic prefill-chunk sweep clean (CPU fills 12-51, GPU 12-31); iPhone 17 Pro composite probe 8/8 on GPU and CPU.
- Hybrid packaging: composite hybrid cache layer (KV + conv + recurrent state at one layer index), folded selective scan as batched matmuls with chunk/head axes in the batch axis (all tensors rank <= 4, no BROADCAST_TO, no int64 index math — what makes the graph fully delegable on GPU), Falcon µP multiplier vector and ssm_in_multiplier preserved, prefill-pad guard (README Conversion notes).
- Galaxy S26 GPU per the README: runs, 115393/115393 ops across 12 subgraphs on LiteRT GPU; this repo's S26 CPU row is from the S7 backfill (device runs).
- License: Falcon LLM License inherited from the base model; changes = weights converted and quantized as stated, tokenizer and chat template repackaged unmodified; community conversion (README License and changes).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
