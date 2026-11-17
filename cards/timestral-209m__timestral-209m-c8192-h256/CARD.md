---
family: timestral-209m
license: other (modified-mit)
model_id: timestral-209m__timestral-209m-c8192-h256
source_url: https://huggingface.co/mlboydaisuke/Timestral-209M-LiteRT
task: time-series-forecasting
---

# timestral-209m__timestral-209m-c8192-h256

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
| **Command** | `python scripts/r4_export.py --bucket 256x8 --rows 2 --norm pow2 --norm-k scripts/normk_fixture.json --attn-k 4 --attn-clamp --dec-only-last --fp16  # in timestral/ of github.com/john-rocky/LiteRT-Models, the zoo sample the card links (github.com/john-rocky/LiteRT-Models timestral/README.md at main ceb8299 ('Rebuilding the graphs')), after hf download mistralai/Timestral-209M config.json model.safetensors --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365; the float16-weight output timestral209m_r2_l256_h8_pow2knormk_fixture_att4c_dl_wfp16.tflite is published as timestral-209m-c8192-h256.tflite` |
| **Quantization** | float16 weights (cast with ai-edge-quantizer 0.9.0); inputs and output float32; made for the GPU's FP16 precision (card 'Files', 'How the graphs were made') |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `timestral-209m-c8192-h256.tflite` | `62a92115fb81e16f7b235d402c74df6175c568540b6c0ac14f354d92567763d7` | 386.278 |

## Performance

No benchmark data yet.

## Delegation (static pre-flight)

Static `edge-lint` verdicts against the delegate compatibility matrix — no runtime execution. Verdicts are only valid for this backend and LiteRT version.

| | |
|---|---|
| **Backend** | gpu_mldrift |
| **LiteRT version** | 2.2.0 |
| **Op coverage** | 85.4% |
| **Partitions** | 276 |
| **Blocking ops** | DEQUANTIZE, LOGISTIC, EXP, LOG |
| **Verdict provenance** | measured=1825, unmatched=58 |
| **Lint report schema** | 1.1 |

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/2.2.0/2026-10-22/timestral-209m__timestral-209m-c8192-h256__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu_xnnpack | pass | yes | pass | sig:CPU XNNPACK 4 threads; gate app com.mlboydaisuke.d1omni.gate (generic mode) | 294.1965365 | - | - | - | 1344.125 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android 16 BP4A.251205.006.S942QOPS1AZH9 | 2026-10-22 | measured |
| galaxy-s26 | gpu_mldrift | pass | yes | - | sig:GpuOptions precision=FP16; gate app com.mlboydaisuke.d1omni.gate (generic mode) | 68.714922 | - | - | - | 1874.531 | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 (Adreno GPU, Hexagon HTP) · litert 2.2.0 · Android 16 BP4A.251205.006.S942QOPS1AZH9 | 2026-10-22 | measured |

## Pitfalls

- Timestral-209M's attention logits (up to 242,875) and RMSNorm sums of squares (up to 164,602,176) leave the float16 range (65,504): each RMSNorm input is pre-scaled by a power of two, the logits are computed at 1/16 with the row maximum subtracted before the scale is restored, the softmax input is clamped at -3*10^4 and the mask value is -3*10^4; in float32 these rewrites leave the results unchanged (card 'How the graphs were made').
- Five float32 inputs, one signature `serving_default`: `patches` [2,256,64] (per patch 32 normalized values then 32 missing flags), `dec` [2,8,768] (`forecast_query.f32` for every decoder token), `bias` [1,1,264,264] (additive mask: 0 or -3*10^4), `cos` / `sin` [1,1,264,64] (rows cut from `rope_cos.f32` / `rope_sin.f32`) -> `output_0` [2,8,32,21], the normalized quantiles; the two rows are the series and its negation (card 'Inputs and output').
- The graph is one step of the forecast: the host builds the inputs and finishes the forecast the way the official forecaster does (causal float64 scaling and asinh, the attention mask, the inverse map and the advancing refine frame, the caps at the context's range widened by 10^4 times the scale, symmetric averaging with the negation, clipping at 0 for non-negative series); the repository's `host_forecast.py` and `android/TimestralHost.kt` do it (card 'Inputs and output').
- A file holds a context of up to 8,192 steps and a horizon of up to 256 steps (patches of 32): the host pads a shorter series in front with missing steps and keeps the requested steps; the smallest file that holds the series and the horizon takes the least time (card 'Files').
- One series per forecast without covariates; a horizon of up to 1,024 steps (the Python class and the Kotlin host stop there; no result covers the official block-by-block decoding of longer horizons); of a series longer than 8,192 steps the last 8,192 are read (the official forecaster with max_context = 8192; its default is 32,768); the Python class refuses fewer than 16 values (card 'Limits').
- The float16 files are made for the GPU's FP16 precision: set it explicitly, `CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)` (card 'Minimal usage').
- The `c8192` files are not used on the NPU: on the Galaxy S26, other forms of these graphs took 82.0 to 96.5 ms per forecast on the NPU against 68.7 to 73.4 ms for these files on its GPU (card 'Speed on phones').

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
