---
family: timestral-209m
license: other (modified-mit)
model_id: timestral-209m__timestral-209m-c8192-h256-int8
source_url: https://huggingface.co/mlboydaisuke/Timestral-209M-LiteRT
task: time-series-forecasting
---

# timestral-209m__timestral-209m-c8192-h256-int8

| | |
|---|---|
| **Task** | time-series-forecasting |
| **Family** | timestral-209m |
| **Source** | https://huggingface.co/mlboydaisuke/Timestral-209M-LiteRT |
| **License** | other (modified-mit) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.9.4 (ai-edge-quantizer 0.9.0 for the weights; torch 2.13.0; ai-edge-litert 2.2.0 for the desktop checks) |
| **Command** | `python scripts/r4_export.py --bucket 256x8 --rows 2 --norm pow2 --norm-k scripts/normk_fixture.json --attn-k 4 --attn-clamp --dec-only-last --fp16 && python scripts/r4_quant.py drq --src out/tflite/timestral209m_r2_l256_h8_pow2knormk_fixture_att4c_dl_fp32.tflite --sel blocks --rest-fp16  # in timestral/ of github.com/john-rocky/LiteRT-Models, the zoo sample the card links (github.com/john-rocky/LiteRT-Models timestral/README.md at main ceb8299 ('Rebuilding the graphs')), after hf download mistralai/Timestral-209M config.json model.safetensors --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365; the output timestral209m_r2_l256_h8_pow2knormk_fixture_att4c_dl_blocksint8w16.tflite is published as timestral-209m-c8192-h256-int8.tflite` |
| **Quantization** | dynamic-range int8 from ai-edge-quantizer 0.9.0 (int8 weights per channel, inputs quantized at run time) on 199 of the 217 fully connected layers, 91.6 % of their weights: all those of the first 23 of the 24 transformer blocks; the other 18 keep float16 weights (3 in the patch embedding, 11 in the last block, 4 in the quantile head); inputs and output float32 (card 'How the graphs were made') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `timestral-209m-c8192-h256-int8.tflite` | `cd4ef83f6104cb381c5dad741063b131da30b8fa10bbf39cdc60358ee8e0b1e6` | 212.037 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 95.5% |
| **Partitions** | 77 |
| **Blocking ops** | LOGISTIC, DEQUANTIZE, EXP, LOG |
| **Verdict provenance** | measured=1626, unmatched=58 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-22/timestral-209m__timestral-209m-c8192-h256-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:CPU XNNPACK 4 threads; gate app com.mlboydaisuke.d1omni.gate (generic mode) | 128.49638 | - | - | - | 645.496 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android 16 BP4A.251205.006.S942QOPS1AZH9 | 2026-10-22 | measured |

## Pitfalls

- Timestral-209M's attention logits (up to 242,875) and RMSNorm sums of squares (up to 164,602,176) leave the float16 range (65,504): each RMSNorm input is pre-scaled by a power of two, the logits are computed at 1/16 with the row maximum subtracted before the scale is restored, the softmax input is clamped at -3*10^4 and the mask value is -3*10^4; in float32 these rewrites leave the results unchanged (card 'How the graphs were made').
- Five float32 inputs, one signature `serving_default`: `patches` [2,256,64] (per patch 32 normalized values then 32 missing flags), `dec` [2,8,768] (`forecast_query.f32` for every decoder token), `bias` [1,1,264,264] (additive mask: 0 or -3*10^4), `cos` / `sin` [1,1,264,64] (rows cut from `rope_cos.f32` / `rope_sin.f32`) -> `output_0` [2,8,32,21], the normalized quantiles; the two rows are the series and its negation (card 'Inputs and output').
- The graph is one step of the forecast: the host builds the inputs and finishes the forecast the way the official forecaster does (causal float64 scaling and asinh, the attention mask, the inverse map and the advancing refine frame, the caps at the context's range widened by 10^4 times the scale, symmetric averaging with the negation, clipping at 0 for non-negative series); the repository's `host_forecast.py` and `android/TimestralHost.kt` do it (card 'Inputs and output').
- A file holds a context of up to 8,192 steps and a horizon of up to 256 steps (patches of 32): the host pads a shorter series in front with missing steps and keeps the requested steps; the smallest file that holds the series and the horizon takes the least time (card 'Files').
- One series per forecast without covariates; a horizon of up to 1,024 steps (the Python class and the Kotlin host stop there; no result covers the official block-by-block decoding of longer horizons); of a series longer than 8,192 steps the last 8,192 are read (the official forecaster with max_context = 8192; its default is 32,768); the Python class refuses fewer than 16 values (card 'Limits').
- The `-int8` files are the CPU files; the CPU runs the float16 files too, more slowly: on the Galaxy S26 CPU (4 threads) `timestral-209m-c8192-h256-int8.tflite` takes 128.5 ms per forecast and `timestral-209m-c8192-h256.tflite` 294.2 ms; most of their fully connected layers compute in int8, so their quantiles differ slightly from float32 (card 'Files', 'Speed on phones', 'Limits').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
