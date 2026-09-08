---
family: granite4h
license: apache-2.0
model_id: granite4h-1b-gpufix
source_url: https://huggingface.co/ibm-granite/granite-4.0-h-1b
task: text-generation
---

# granite4h-1b-gpufix

| | |
|---|---|
| **Task** | text-generation |
| **Family** | granite4h |
| **Source** | https://huggingface.co/ibm-granite/granite-4.0-h-1b |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch, pinned base + granite hybrid patch, selective scan folded to batched matmuls (all tensors rank <= 4, no giant broadcast intermediates) — hf-to-litertlm granite_work/convert_granite4h.py via gpu_ship_20260813/build_1b_final.sh editable checkout 115a136 + granite_hybrid_litert_torch.patch (hf-to-litertlm granite_work/README.md; folded-scan revision of 2026-08-13) |
| **Command** | `PATH=$REPO/.venv-092/bin:$PATH PYTHONPATH=$HOME/code/litert-torch .venv-092/bin/python $HOME/code/hf-to-litertlm/granite_work/convert_granite4h.py ibm-granite/granite-4.0-h-1b <outdir>  # gpu_ship_20260813/build_1b_final.sh verbatim` |
| **Quantization** | int8 dynamic on linears + embedding; convs and the selective scan stay float; fp32 activations declared (GPU requirement — HF card update note) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `granite-4.0-h-1b_int8_fp32act.litertlm` | `c4c5d09334b974abd2f80893b1b46df036a8ff28b54e70d36309326f9f50fa06` | 1605.158 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-13/granite4h-1b-gpufix__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel-8a | cpu | pass | - | - | 40 | - | 5.14 | 4.34 | 8010.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |
| pixel-8a | gpu | pass | yes | - | 40 | - | 23.9 | 8.81 | 1790.0 | - | pixel-8a · Google Tensor G3 · litert-lm 0.16.0 | 2026-08-13 | measured |

## Pitfalls

- GPU execution requires fp32 activations (declared in this bundle) — the memory cost is stated on the HF card's honest notes; CPU decode is ~3.5x the previous file (11.4 -> 39.2 tok/s) from the folded scan alone (HF card update, measured).
- Full GPU delegation on Android OpenCL, one partition per subgraph, zero rejections (4628/4628, 2240/2240, 2271/2271) — litertlm-convert commit 2cd79bf; end-to-end GPU verified on macOS, iPhone 17 Pro (Metal) and Pixel 8a per the HF card update.
- First Mamba2-hybrid selective scan on a mobile GPU (HF card claim of record).
- Family BOS trap unchanged from the 08-04 lineage: the 350m sibling's greedy flips with BOS (LlmMetadata start_token); the 1b is BOS-robust (commit 5f3176d).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
