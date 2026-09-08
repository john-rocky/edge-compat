---
family: phi
license: mit
model_id: phi4-mini-reasoning
source_url: https://huggingface.co/litert-community/Phi-4-mini-reasoning
task: text-generation
---

# phi4-mini-reasoning

| | |
|---|---|
| **Task** | text-generation |
| **Family** | phi |
| **Source** | https://huggingface.co/litert-community/Phi-4-mini-reasoning |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (official converter; Phi3ForCausalLM with LongRoPE + a nominal sliding window, two export-time adjustments) (HF card Conversion) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `TODO (the HF card states the two adjustments and the recipe, not the invocation)` |
| **Quantization** | int4 blockwise-32 + OCTAV, embeddings INT8, KV cache 4096, externalize_embedder=True (HF card Conversion) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model.litertlm` | `ee46dec376e1aee8190c41bce5f29d3b891db3eaa0e92ce5a6a28e24f04067ba` | 2655.005 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.14.0/2026-07-23/phi4-mini-reasoning__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-01/phi4-mini-reasoning__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 1135.3 | 83.68 | 237.4 | - | Mac Studio (M4 Max) · Apple M4 Max · litert-lm 0.14.0 | 2026-07-23 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 14.06 | 1.62 | 21144.8 | 4274.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Two export-time adjustments for current litert-torch: (1) LongRoPE — replace Phi3RotaryEmbedding.forward with a static version (the @dynamic_rope_update seq-len branch is data-dependent under torch.export; for cache <= original_max=4096 the short factor is always correct); (2) sliding window — set config.sliding_window=None (it is 262144 >> context, i.e. full-causal) so the standard causal mask path is used (HF card Conversion).
- License MIT, inherited from microsoft/Phi-4-mini-reasoning (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
