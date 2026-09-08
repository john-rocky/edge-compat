---
family: qwen3.5
license: apache-2.0
model_id: qwen35-4b
source_url: https://huggingface.co/Qwen/Qwen3.5-4B
task: text-generation
---

# qwen35-4b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3.5 |
| **Source** | https://huggingface.co/Qwen/Qwen3.5-4B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + Qwen3.5 hybrid patch (same v4 rank-<=4 chunk kernel as the 0.8B) + two 4B-specific additions — a rank-4 head-interleave for grouped value heads, and a reduced prefill-signature ladder editable checkout 115a136 + qwen35_hybrid_litert_torch.patch (regenerated 2026-08-14 with the interleave fix; applies clean to pristine 115a136) |
| **Command** | `QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 python convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_ship_4b  # then scripts/set_activation_type.py --type fp32; export log qwen35_work/convert_ship_4b_v3.log` |
| **Quantization** | int8 dynamic on linears + embedding (wi8fc); convs and the delta rule float; fp32 activations declared (GPU requirement) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-4B_int8.litertlm` | `f0abbbc69b4126ddcb208a75e8904ecadff87aa24202e3de6ffd518c198af1ec` | 4203.251 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-14/qwen35-4b__mac-m4-max.json`, `data/device_runs/0.16.1/2026-09-01/qwen35-4b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-m4-max | cpu | pass | - | - | 256 | - | 242.61 | 19.9 | 1105.5 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| mac-m4-max | gpu | pass | - | - | 256 | - | 672.16 | 61.98 | 397.0 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 20.24 | 1.83 | 13196.2 | 6368.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Grouped value heads (32 v / 16 k — first Qwen3.5 with ratio > 1) trace upstream's repeat_interleave(r, dim=2); its rank-5 unsqueeze->expand->reshape lowering is rejected by the GPU delegate ('RESHAPE: Tensor dimensions must be less than 5') and engine creation aborts. Ratio-1 checkpoints (0.8B/2B) never trace the branch, so the family's smaller models hide the wall. Fixed in-patch: fold batch, concat copies, rank-4 reshape — bitwise-identical (commit 8ecfbbe).
- Signature-count RAM law hits at 4B scale: the full 11+1 prefill ladder jetsam-kills a 12 GB iPhone at Metal program-init 9/12 (248k-vocab per-signature programs; Nemotron-H-4B with the same 12 sigs fits at 6.69 GB peak). Shipped with a 7-signature ladder -> iPhone GPU peak 4.98 GB. A ladder change alters the runtime's chunk plans, so the prompt-length sweep must re-run (it did: 40/40 CPU + 20/20 GPU).
- fp32 activations mandatory on GPU (family requirement, declared in the bundle TOML).
- Gates on the shipped file: FP parity 48 positions top-1/top-5 100% / Pearson 1.0000 / KL ~0 (split harness scripts/parity_logits_bigmodel.py — the 16 GB fp32 tflite exceeds the Python Interpreter's flatbuffer limit, so pt/lt stages run in separate venvs over a decode+p32,16 dev export); 8Q 8/8 on CPU and GPU on litert-lm 0.15.0 AND 0.16.0; BANANA hermetic CPU 40/40 + GPU 20/20; iPhone 17 Pro Metal composite 8-question probe word-for-word identical to HF fp32 (9.28 tok/s decode / 87.1 prefill / TTFT 1.93 s / peak 4975 MB; CPU 5.43 / 24.0 / 1703 MB). Mac bench (0.16.0, cache no, p256/d256, quiet): GPU 672.16/61.98 TTFT 0.40 s vs CPU 242.61/19.90 TTFT 1.11 s.
- Honest quality note (HF card): on the composite 8-question probe the CPU int8 path answers 6 of 8 (every given answer correct; HF fp32 itself answers 8/8 — an int8-on-CPU cost); single-question gates are 8/8 on every backend. GPU is the fidelity path for this file.
- Android: 8 GB-class phones cannot fit the 4.1 GB file (Pixel 8a not attempted — 3.3 GB free vs 4.1 GB); Apple-hardware-first release.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
