---
family: zamba2
license: apache-2.0
model_id: zamba2-2.7b-instruct
source_url: https://huggingface.co/litert-community/Zamba2-2.7B-instruct
task: text-generation
---

# zamba2-2.7b-instruct

| | |
|---|---|
| **Task** | text-generation |
| **Family** | zamba2 |
| **Source** | https://huggingface.co/litert-community/Zamba2-2.7B-instruct |
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
| `Zamba2-2.7B-instruct_int8.litertlm` | `601e96057ddeb6fc1a7451f193e475419c7b121408d841c08cef391f051835f1` | 2673.251 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-09-02/zamba2-2.7b-instruct__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 14.64 | 1.03 | 18457.7 | 4996.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- iPhone 17 Pro: the 8-item composite probe answers 8/8 on the CPU backend; the Metal backend does not run this size on a 12 GB phone (README Correctness + honest note).
- Galaxy S26 GPU per the README: does not run — engine creation rebooted the phone (README S26 table); this repo carries the Pi 5 CPU row (device runs).
- Conversion-side note: current transformers (5.14.x and main as of 2026-08-14) cannot load ANY two-block Zamba2 checkpoint (2.7B/7B) — Zamba2Model.get_layers assigns block_id by global layer index while the checkpoint layout and its weight-tie cycle follow hybrid occurrence order; the conversion assigns block_id by hybrid occurrence order (README Correctness + Conversion notes).
- Requires litert-lm >= 0.15 (README intro).
- Correctness (README): float export matches the HF model teacher-forced across 48 decode positions (top-1/top-5 identical, corr 1.0000, KL ≈ 0); 8-question gate 8/8 on every lane (GPU and CPU, litert-lm 0.15.0 and 0.16.0), no degeneration, no greedy flips; hermetic prefill-chunk sweep clean (CPU fills 12-51, GPU 12-31).
- Hybrid packaging: folded selective scan (batched matmuls, all tensors rank <= 4 — what makes the graph fully delegable on GPU), min-only dt clamp handling (padded prefill positions forced to exact identity steps AFTER the clamp), shared block + adapters traced once per position with tied weights stored once, composite hybrid cache layer, prefill-pad guard (README Conversion notes).
- Streaming detokenization: the tokenizer's Strip decoder is removed from the bundle — Zamba2's metaspace (SP-BPE) tokenizer otherwise loses every interior space under the runtime's per-token streaming decode; the only behavior change is a sequence-initial space, which the runtime trims (README Conversion notes).
- License Apache-2.0, inherited from the Zyphra base (README front matter).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
