---
family: qwen3.5
license: apache-2.0
model_id: qwen35-4b-mixed-int4
source_url: https://huggingface.co/Qwen/Qwen3.5-4B
task: text-generation
---

# qwen35-4b-mixed-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3.5 |
| **Source** | https://huggingface.co/Qwen/Qwen3.5-4B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + Qwen3.5 hybrid patch (identical float export rail to qwen35-4b int8), then post-hoc ai-edge-quantizer via qwen35_work/make_int4_4b.py editable checkout 115a136 + qwen35_hybrid_litert_torch.patch; ai-edge-quantizer from the ltconv040dev venv |
| **Command** | `QWEN35_PREFILL_LADDER=1024,256,64,16,4,1 python convert_qwen35_hybrid.py Qwen/Qwen3.5-4B out_int4_4b  # float export; then: python make_int4_4b.py out_int4_4b/model.litertlm out_int4_4b b32` |
| **Quantization** | Mixed INT4: int4 BLOCKWISE-32 min-max on every FULLY_CONNECTED, int8 channelwise on lm_head + embedding (one shared 248320x2560 vocab buffer, verified single); convs and the delta rule float; fp32 activations declared |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen3.5-4B_mixed_int4.litertlm` | `176b5a2b6c20bde7739fba98e653a04f5d5b4a753a4b9dd0ff529aba7c35be99` | 2626.768 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-27/qwen35-4b-mixed-int4__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-27/qwen35-4b-mixed-int4__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-08-27/qwen35-4b-mixed-int4__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-01/qwen35-4b-mixed-int4__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iphone-17-pro | cpu | pass | - | - | - | - | 46.68 | 8.87 | 3124.0 | 1678.757 | iphone-17-pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-27 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | 88.69 | 11.38 | 1894.0 | 5735.633 | iphone-17-pro · A19-Pro · litert-lm 0.16.0 · iOS 27.0 | 2026-08-27 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 100.08 | 20.07 | 2607.8 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 669.33 | 68.45 | 397.1 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| pixel-8a | cpu | fallback | no | - | 260 | - | 22.13 | 6.3 | 11910.0 | - | pixel-8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-08-27 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 17.08 | 2.37 | 15416.4 | 4969.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Block size chosen by measurement, not family habit: b32/b128 x min-max/OCTAV all built from the SAME float export and scored on GSM8K n=100 (greedy, 2048-token budget, litert-mac-verify GPU): int8 control 97, b32 min-max 93 (ships), b32 OCTAV 92, b128 min-max 90, b128 OCTAV 90. OCTAV is a no-op on this model; blockwise-128 (the dense-4B habit) costs 3 points on this hybrid.
- GSM8K at a 512-token budget is a false-fail generator on this family even with thinking disabled: the verbose CoT truncates before '#### N' and the extractor grabs stray numbers — it reads as quantization collapse and is actually truncation (measured; the same questions come back correct at 2048).
- The iPhone b32-vs-b128 decode gap is GONE on this runtime (CLiteRTLM v0.16.0: 11.38 vs 11.85 tok/s Metal — parity); the historical ~2x b128 edge from dense ships is a dated kernel observation, not a law.
- The 8Q and prompt-length gates cannot see the int4 cost at all (every cell 8/8 / clean on both variants); only GSM8K n=100 separates the recipes.
- Gates on the shipped file: 8Q 8/8 on CPU and GPU on litert-lm 0.15.0 AND 0.16.0; BANANA hermetic CPU 40/40 + GPU 20/20; iPhone 17 Pro composite 8-question probe 8/8 on BOTH Metal and CPU (the int8 sibling's CPU path answers 6/8) — Metal 11.38 tok/s decode / 88.7 prefill / TTFT 1.89 s / peak 5736 MB, CPU 8.87 / 46.7 / 3.12 s / 1679 MB, cold, unplugged. Mac bench (0.16.0, cache no, p256/d256, quiet, 300 s GPU rest): GPU 669.33/68.45 TTFT 0.40 s vs CPU 100.08/20.07 TTFT 2.61 s.
- Mac CPU prefill is ~2.4x SLOWER than the int8 sibling (100 vs 243 tok/s; TTFT 2.61 vs 1.11 s) — int4 unpack cost in XNNPACK prefill; decode is equal or faster everywhere. iPhone CPU is the exception (prefill 46.7 vs int8's 24.0 — faster).
- Pixel 8a (8 GB) CPU: FITS and generates coherently — VmHWM 4.43 GB, prefill 22.1 / decode 6.30 tok/s (260-token prompt / 204 decoded), TTFT 11.9 s, engine init 32-48 s, XNNPACK delegates 32830/34373 nodes (litert_lm_main android_arm64 v0.16.0, warm weight cache). GPU on 8 GB devices NOT attempted — engine creation is the known device-killer (2B lesson).
- First Android CPU run writes a 2.66 GB XNNPACK weight cache next to the model. The cache key embeds the model file mtime: two runs that see mtimes 1 s apart build TWO 2.66 GB caches — check `ls *.xnnpack_cache | wc -l` after first runs.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
