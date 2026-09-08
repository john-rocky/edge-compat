---
family: lfm2.5
license: lfm1.0
model_id: lfm25-encoder-230m
source_url: https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M
task: feature-extraction
---

# lfm25-encoder-230m

| | |
|---|---|
| **Task** | feature-extraction |
| **Family** | lfm2.5 |
| **Source** | https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M |
| **License** | lfm1.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (litertlm-convert scripts/convert_lfm25_encoder.py, litert_torch.signature multi-signature trace) 0.9.2 (stock release; .venv-092 per lfm25_encoder_work/FINDINGS.md) |
| **Command** | `python scripts/convert_lfm25_encoder.py LiquidAI/LFM2.5-Encoder-230M out/lfm25_encoder_230m` |
| **Quantization** | int8 dynamic-range (linears + embedding; convs float) — wi8fc variant |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `LFM2.5-Encoder-230M_wi8fc.tflite` | `6be33404190793ec1bee8b054bb0903e2bf5a33b266aa224a96c74a829247743` | 234.397 |

## Performance

No benchmark data yet.

## Pitfalls

- The fp16 variant is desktop-oriented: XNNPACK's per-signature fp32 unpacking is heavy on phone memory limits — use the int8 (wi8fc) file on mobile (iPhone-verified bit-exact).
- All signatures are batch-1, right-padded static shapes (encode_64/128/256/512, mlm_128); padded positions are fully masked in-graph, so outputs at valid positions are independent of padding length.
- iPhone 17 Pro int8 run reproduces Mac outputs bit-exactly, but peak footprint is ~1.0 GiB.
- License: LFM Open License v1.0 — note the commercial-use threshold (Section 5); redistributed as Derivative Works with modification notices per Section 4.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
