---
family: qwen2.5-coder
license: apache-2.0
model_id: qwen25-coder-1.5b-int4
source_url: https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct
task: text-generation
---

# qwen25-coder-1.5b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | qwen2.5-coder |
| **Source** | https://huggingface.co/litert-community/Qwen2.5-Coder-1.5B-Instruct |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (hf-to-litertlm `qwen25coder_work/convert_qwen25_coder.sh`; export_simple_template.py) litert-torch 0.9.3 (litert-converter 0.3.1, ai-edge-quantizer 0.8.0, litert-lm-builder 0.16.0) — stated on the published HF card |
| **Command** | `CACHE=4096 PREFILL=1024,256,64,16,4,1 EXTERNALIZE_EMBEDDER=1 python scripts/export_simple_template.py src_models/qwen25-coder-1.5b out_qwen25_coder qwen25coder_work/qwen25_coder_simple.jinja BOCTAV4` |
| **Quantization** | int4 weights, blockwise-32 + OCTAV, symmetric (BOCTAV4) on linears; int8 embedding, externalised (tied lm_head) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Qwen2.5-Coder-1.5B-Instruct_int4.litertlm` | `273ecc7771ba2dd5fe1bb6d4d4726ad0353102f04ad094082ccf59bca9f21213` | 1065.622 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-19/qwen25-coder-1.5b-int4__pixel-8a.json`, `data/device_runs/0.16.0/2026-08-24/qwen25-coder-1.5b-int4__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-05/qwen25-coder-1.5b-int4__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/qwen25-coder-1.5b-int4__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | 223 | - | 106.0 | 29.73 | 2140.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | pass | yes | - | 223 | - | 205.89 | 10.07 | 1180.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| pixel-8a | cpu | pass | - | - | 38 | - | 16.09 | 12.57 | 2440.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-19 | measured |
| pixel-8a | gpu | pass | yes | - | 38 | - | 83.7 | 12.0 | 540.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 | 2026-08-19 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 41.69 | 5.61 | 7269.2 | 1842.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- EXTERNALIZE_EMBEDDER=1 is required, not cosmetic: the model ties embedding and lm_head, and BOCTAV4 (int4 linears + int8 embedding) describes one tensor two ways — the quantizer resolves it by copying the 151936x1536 table once per prefill signature (7 copies, 1.63 GB, 65% of the file, no error). Externalising removes the conflict (HF card / REPRODUCE.md).
- The upstream chat template injects a default system prompt ('You are Qwen, created by Alibaba Cloud…') when none is supplied; a generic ChatML template omits it and runs the model outside its tuned state — the bundle carries a template that reproduces the default (FINDINGS.md / REPRODUCE.md).
- Requires litert-lm >= 0.16 (HF card).
- Pixel 8a 2026-08-19: fully delegated on Mali OpenCL (1243/1243 per prefill signature, 1132/1132 decode, zero rejections); GPU buys prefill/TTFT (0.54 s vs 2.44 s on CPU), not decode (12.0 vs 12.6 tok/s — same LPDDR) (FINDINGS.md).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
