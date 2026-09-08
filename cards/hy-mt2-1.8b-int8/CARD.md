---
family: hy-mt2
license: apache-2.0
model_id: hy-mt2-1.8b-int8
source_url: https://huggingface.co/litert-community/Hy-MT2-1.8B
task: text-generation
---

# hy-mt2-1.8b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | hy-mt2 |
| **Source** | https://huggingface.co/litert-community/Hy-MT2-1.8B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch via hf-to-litertlm (`python scripts/convert.py tencent/Hy-MT2-1.8B`, one command; export 120 s on an M4 Max) (HF card Conversion notes) 0.9.3 (stock litert-torch — HF card Conversion notes) |
| **Command** | `python scripts/convert.py tencent/Hy-MT2-1.8B (hf-to-litertlm)` |
| **Quantization** | export-time dynamic int8 on linears + embedding (HF card file table) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Hy-MT2-1.8B_int8.litertlm` | `7bbd0b65d7e69c8d92b8ab3033e353536ef40c77f4dc5f4c32f2931417c13eda` | 1731.513 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-27/hy-mt2-1.8b-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/hy-mt2-1.8b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/hy-mt2-1.8b-int8__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-02/hy-mt2-1.8b-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 197 | - | 126.34 | 12.27 | 1640.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 197 | - | 385.67 | 20.84 | 560.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 210.95 | 33.97 | 1243.1 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 2008.37 | 105.8 | 136.9 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| pixel-8a | cpu | fallback | no | - | 197 | - | 50.78 | 8.3 | 4000.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 25.01 | 2.66 | 10613.9 | 2950.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Translation model, not open chat: use the source card's default translation prompt verbatim; the 8-question sanity gate reads 6/8 on CPU and 6/8 on GPU with the same two misses on both backends ('Cool' for the opposite of hot, 'pink' for the rhyme) — a property of this translation-tuned 1.8B, not of a backend; arithmetic, factual and translation items are correct (HF card Correctness, litert-lm 0.16.0, M4 Max).
- Chat template byte-equal to the source repo's chat_template.jinja (654/654 bytes); stop token <|hy_place_holder_no_2|> (id 120020) as the source declares (HF card Correctness).
- No duplicate start token: the source template renders <|hy_begin_of_sentence|> itself and the engine also prepends the metadata start_token; measured inside the runtime, [start_token]+prompt and [template BOS]+prompt generate byte-identical greedy output (HF card Correctness + Conversion notes).
- The 'dynamic'-with-'alpha' rope resolves statically: transformers computes base = rope_theta * alpha^(dim/(dim-2)) once at init and never rescales below max_position_embeddings; only the data-dependent cache-growth branch kills torch.export, and the converter bakes the resolved base (HF card Conversion notes).
- Translation greedy A/B vs the HF bf16 reference: byte-identical on 1 of 3 probes, fluent alternates on the other two (the usual int8-vs-bf16 kind), no degeneration (HF card Correctness).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/ (Pi 5 / S26 / Pixel 8a / Mac / iPhone as measured).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
