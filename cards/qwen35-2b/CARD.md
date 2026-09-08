---
family: qwen3_5
license: apache-2.0
model_id: qwen35-2b
source_url: https://huggingface.co/Qwen/Qwen3.5-2B
task: text-generation
---

# qwen35-2b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3_5 |
| **Source** | https://huggingface.co/Qwen/Qwen3.5-2B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + Qwen3.5 hybrid patch (same v4 rank-<=4 chunk kernel as the 0.8B/4B); no model-specific additions — the 2B trips neither 4B wall editable checkout 115a136 + qwen35_hybrid_litert_torch.patch (unchanged from the 4B regeneration of 2026-08-14) |
| **Command** | `python convert_qwen35_hybrid.py Qwen/Qwen3.5-2B out_ship_2b  # full default prefill ladder; then scripts/set_activation_type.py --type fp32; export log qwen35_work/convert_ship_2b.log` |
| **Quantization** | int8 dynamic on linears + embedding (wi8fc); convs and the delta rule float; fp32 activations declared (GPU requirement) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-2B_int8.litertlm` | `86c97baba6d3fb4109588562f0b9411e502c883be7fc2479f93ce3a6ea3efb6b` | 2018.54 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-14/qwen35-2b__mac-m4-max.json`, `data/device_runs/0.16.1/2026-09-01/qwen35-2b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-m4-max | cpu | pass | - | - | 256 | - | 592.1 | 37.6 | 461.7 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| mac-m4-max | gpu | pass | - | - | 256 | - | 1485.71 | 114.34 | 181.1 | - | mac-m4-max · litert-lm 0.16.0 | 2026-08-14 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 55.1 | 4.25 | 4881.2 | 3863.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- The 2B is the zero-extra-work sibling: linear-attention heads are ungrouped (16 k / 16 v, ratio 1), so the 4B's repeat_interleave rank-5 wall never traces, and the full 11+1-signature prefill ladder fits a 12 GB iPhone (GPU peak 5.33 GB during engine creation vs the 4B's jetsam at Metal program-init 9/12 — the 248k-vocab per-signature cost is weight-size-dependent).
- fp32 activations mandatory on GPU (family requirement, declared in the bundle TOML).
- Gates on the shipped file: FP parity 48 positions top-1/top-5 100% / Pearson 1.0000 / KL ~0 (split harness scripts/parity_logits_bigmodel.py over a decode+p32,16 dev export); 8Q 8/8 on CPU and GPU on litert-lm 0.15.0 AND 0.16.0; BANANA hermetic CPU 40/40 + GPU 20/20; iPhone 17 Pro Metal composite 8-question probe word-for-word identical to HF fp32 through answer 7 — including the model's own arithmetic slip on Q1 — then int8 diverges by one end-of-turn token and adds a correct 8th answer (24.33 tok/s decode / 237.7 prefill / TTFT 0.73 s / peak 5330 MB; CPU 16.17 / 206.5 / TTFT 0.77 s / peak 1522 MB). Mac bench (0.16.0, cache no, p256/d256, quiet): GPU 1485.71/114.34 TTFT 0.18 s vs CPU 592.1/37.6 TTFT 0.46 s.
- Honest quality note (HF card): on the composite 8-question probe the CPU int8 path degrades hard at 2B scale — 3 of 8 answers, deterministically identical on Mac and iPhone (backend property, not device flakiness); single-question gates are 8/8 on every backend and both runtime versions. GPU is the fidelity path for this file.
- Pixel 8a (8 GB Android): GPU delegation is clean (zero rejections, 16865/16865 nodes on every compiled signature) but engine creation OOMs the phone — it rebooted mid-init; the fp32-expanded weight buffers exceed device RAM. CPU is separately blocked on this 97%-full phone: the XNNPACK weight cache needs ~another file-size of free disk and aborts with ENOSPC; --disable_weight_cache, --disable_cache=true and --cache_dir=:memory all fail to prevent the cache write on the pinned v0.16.0 litert_lm_main binary (new trap, 2026-08-14). Apple-hardware-first release.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
