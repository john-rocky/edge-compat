---
family: qwen3
license: mit
model_id: fastcontext-1.0-4b-sft-model-block128
source_url: https://huggingface.co/litert-community/FastContext-1.0-4B-SFT
task: text-generation
---

# fastcontext-1.0-4b-sft-model-block128

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen3 |
| **Source** | https://huggingface.co/litert-community/FastContext-1.0-4B-SFT |
| **License** | mit |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (official converter; FastContext is a standard Qwen3ForCausalLM, existing Qwen3 path, no custom graph code) (HF card Conversion) TODO (the HF card names the converter but not its version; no export log in the sources) |
| **Command** | `from litert_torch.generative.export_hf.export import export; export(model='microsoft/FastContext-1.0-4B-SFT', output_dir='out', quantization_recipe='qwen3_int4_block32_octav.json' | 'qwen3_int4_block128_octav.json', cache_length=4096, externalize_embedder=True) (HF card Conversion snippet)` |
| **Quantization** | int4 blockwise (block 128) + OCTAV, symmetric; INT8 embedding externalized; KV cache 4096 (HF card) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `model_block128.litertlm` | `3842a564a1f8db2ed53992ec6e2d5e2579aff72131dfac664c19cf2ec90542ce` | 2359.731 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/fastcontext-1.0-4b-sft-model-block128__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-02/fastcontext-1.0-4b-sft-model-block128__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 202 | - | 86.28 | 7.49 | 2470.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 10.33 | 1.49 | 26039.6 | 4009.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Which file: block 32 (model.litertlm) is the quality pick — GSM8K 88.0% vs 81.0% for block 128 — while block 128 (model_block128.litertlm) decodes ~40% faster on the iPhone 17 Pro (14 vs 10 tok/s; Mac M-series GPU 66-68 vs 64-73 tok/s) (HF card 'Which file?').
- Blockwise (not the tool's default channelwise) int4 is what preserves accuracy; embeddings kept at INT8 and externalized into their own section (dedups the tied matrix) (HF card Conversion).
- License MIT, inherited from microsoft/FastContext-1.0-4B-SFT (itself built on Qwen3-4B-Instruct) (HF card License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
