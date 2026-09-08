---
family: lfm2.5
license: other (lfm-open-license-v1.0)
model_id: lfm2.5-2.6b-int4
source_url: https://huggingface.co/litert-community/LFM2.5-2.6B
task: text-generation
---

# lfm2.5-2.6b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/litert-community/LFM2.5-2.6B |
| **License** | other (lfm-open-license-v1.0) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (released 0.9.2 with upstream lfm2 hybrid support incl. the ShortConv prefill-pad fix), packaged for litert-lm >= 0.14 with the executor-metadata section binding the 22 conv states + 16 KV caches (HF card Conversion notes) 0.9.2 (HF card Conversion notes) |
| **Command** | `TODO (the HF card states the recipe and post-steps, not the invocation)` |
| **Quantization** | int4 blockwise-32 + OCTAV on linears, int8 embedding, convs float; the checkpoint is sparse — OCTAV int4-b32 produced 746k all-zero weight blocks whose zero scales XNNPACK rejects at load, patched to the tensor's smallest nonzero scale (dequantization unchanged) (HF card file table + Conversion notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-2.6B_int4.litertlm` | `f4797d5a16812231deb6d505b4f417dd835b9b3dc47bb74d9ec71facfd0c91a4` | 1590.873 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/lfm2.5-2.6b-int4__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/lfm2.5-2.6b-int4__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 206 | - | 297.42 | 25.31 | 730.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 28.49 | 4.91 | 9189.5 | 2965.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.14: the bundled executor-metadata section binds the 22 conv states + 16 KV caches (files exported without it do not run) and makes the same file run on 0.15's new state binding (HF card intro + Conversion notes).
- The generation prompt pre-fills <think> exactly as the vendor chat template does; with a bare assistant prompt, think-block emission becomes the model's choice, and int4 degrades that discipline first (unscaffolded rambling in place of answers on multi-turn) (HF card Conversion notes).
- GSM8K (greedy, 0-shot CoT, max-tokens 2048, n=100, same harness for all rows): bf16 reference 92%, int8 88%, int4 83%; both files pass an 8-question sanity gate 8/8 with zero degenerate outputs (CPU), a 42-length prefill sweep with zero corrupt first tokens, and a 3-turn conversation gate (HF card file table + Quality).
- iPhone 17 Pro (CPU backend, cold first runs): the int4 file answers the 8-question gate 8/8 and decodes ~20 tok/s warm; the int8 file loads and runs only when the host app carries the extended-virtual-addressing / increased-memory entitlements (its 2.87 GB single weight section exceeds a default-entitlement app's mmap) — int4 is the recommended phone variant (HF card Speed).
- Multi-length prefill signatures (1-1024) are exported so the runtime picks tight chunks (HF card Conversion notes).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
