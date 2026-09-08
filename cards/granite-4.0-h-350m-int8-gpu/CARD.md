---
family: granite-4.0-h
license: apache-2.0
model_id: granite-4.0-h-350m-int8-gpu
source_url: https://huggingface.co/ibm-granite/granite-4.0-h-350m
task: text-generation
---

# granite-4.0-h-350m-int8-gpu

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite-4.0-h |
| **Source** | https://huggingface.co/ibm-granite/granite-4.0-h-350m |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (granite ext, folded-SSD rewrite) litert-torch 0.10.0 (the checkout's in-tree next-version constant, not a release): ~/code/litert-torch at upstream 115a136, reached via PYTHONPATH (build_350m_gpu.sh:22), which shadows .venv-092's installed 0.9.2 — and the granite ext is uncommitted working-tree state there. Build venv .venv-092, py3.12.13: litert-lm / litert-lm-builder 0.15.0, ai-edge-quantizer 0.8.0, litert-converter 0.3.0. Re-export 2026-08-27. |
| **Command** | `PATH=$REPO/.venv-092/bin:$PATH PYTHONPATH=$HOME/code/litert-torch $REPO/.venv-092/bin/python $PUB/granite_work/convert_granite4h.py ibm-granite/granite-4.0-h-350m $HERE/out_350m && $REPO/.venv-092/bin/python $PUB/granite_work/drop_start_token.py $HERE/out_350m/granite-4.0-h-350m_int8.litertlm $HERE/granite-4.0-h-350m_int8_nobos.litertlm && $REPO/.venv-092/bin/python $REPO/scripts/set_activation_type.py $HERE/granite-4.0-h-350m_int8_nobos.litertlm $HERE/granite-4.0-h-350m_int8_gpu.litertlm --type fp32  # granite_work/gpu_reexport_20260827/build_350m_gpu.sh:30,39,43 verbatim (step 1 guarded by an existence check); REPO=~/code/litertlm-convert, PUB=~/code/hf-to-litertlm, HERE=$REPO/granite_work/gpu_reexport_20260827` |
| **Quantization** | int8 (same weights as the published _int8; re-export changes the scan graph, not the quant) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-h-350m_int8_gpu.litertlm` | `f85136cfd308676e8da5182ca4f29a5d4d63d9bb3d230d9c6e837d5b39a05086` | 458.926 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.15.0/2026-08-28/granite-4.0-h-350m-int8-gpu__iphone-17-pro.json`, `data/device_runs/0.16.0/2026-08-27/granite-4.0-h-350m-int8-gpu__galaxy-s26.json`, `data/device_runs/0.16.0/2026-09-07/granite-4.0-h-350m-int8-gpu__pixel-8a.json`, `data/device_runs/0.16.1/2026-09-02/granite-4.0-h-350m-int8-gpu__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | pass | - | - | 224 | - | 296.78 | 95.16 | 770.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-27 | measured |
| galaxy-s26 | gpu | pass | yes | - | 224 | - | 484.0 | 47.87 | 480.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-27 | measured |
| iphone-17-pro | cpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-28 | measured |
| iphone-17-pro | gpu | pass | - | - | - | - | - | - | - | - | iPhone 17 Pro · A19-Pro · litert-lm 0.15.0 · iOS 27.0 | 2026-08-28 | measured |
| pixel-8a | cpu | fallback | no | - | 224 | - | 110.59 | 40.7 | 2050.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| pixel-8a | gpu | pass | yes | - | 224 | - | 180.07 | 20.7 | 1290.0 | - | Pixel 8a · Tensor G3 · litert-lm 0.16.0 · Android 16 | 2026-09-07 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 88.4 | 14.04 | 2967.4 | 1174.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- The plain _int8/_fp16 files FAIL GPU delegation on Adreno at 109/3673 ops (mamba layer rank-5/6 intermediates; 'bad input dims size: 6', invalid TRANSPOSE). This _int8_gpu re-export writes the Mamba2 selective scan as rank<=4 batched matmuls and declares fp32 activations; on the Galaxy S26 it delegates in full (3512/3512 x12 subgraphs, litert-lm 0.16.0) (gpu_reexport_20260827/RESULTS.md; device_runs 2026-08-27).
- The 'official granite-4.0-350m PASSes so this is a graph-shape difference' framing is wrong and was retracted: that repo is an all-attention dense reinterpretation with zero mamba layers. The valid contrast is h-350m vs h-1b, same architecture — h-1b PASSes on the same folded scan (RESULTS.md, s5_candidates.md correction).
- It is ~45 MB larger than _int8 because each short prefill signature is padded to a full 256-token chunk with a stored constant instead of a PAD op — the trade that lets the graph delegate (published HF card).
- Engine-reuse band: reusing one Engine across conversations with a growing shared prefix can stop replies early; on _int8 this appears at chat-templated lengths 33-37, on _int8_gpu at 37-41 (the folded scan changes the graph, so the band moves). Fresh Engine per conversation is clean at every tested length (LiteRT-LM#3165; published HF card; 92139d3).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
