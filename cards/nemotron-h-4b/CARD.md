---
family: nemotron-h
license: other (nvidia-open-model-license)
model_id: nemotron-h-4b
source_url: https://huggingface.co/nvidia/Nemotron-H-4B-Instruct-128K
task: text-generation
---

# nemotron-h-4b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nemotron-h |
| **Source** | https://huggingface.co/nvidia/Nemotron-H-4B-Instruct-128K |
| **License** | other (nvidia-open-model-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch pinned base + nemotron_h hybrid patch (hf-to-litertlm nemotron_h_work/convert_nemotron_h.py; folded rank<=4 Mamba2 scan ported to NemotronH's SSD spelling, state-continuation tracing, prefill-pad guard) editable checkout 115a136 + nemotron_h_litert_torch.patch |
| **Command** | `git -C litert-torch-nemotron checkout 115a136 && git apply nemotron_h_litert_torch.patch && PYTHONPATH=litert-torch-nemotron python convert_nemotron_h.py nvidia/Nemotron-H-4B-Instruct-128K out_nemotron_4b; then scripts/set_activation_type.py ... --type fp32` |
| **Quantization** | int8 dynamic on linears + embedding; convs and the selective scan stay float; fp32 activations declared in-bundle for GPU |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Nemotron-H-4B-Instruct-128K_int8.litertlm` | `29eee089a631069e5e1b1d1165cbab37570d6cff58efdb488827de2905e54f9c` | 4584.1 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.1/2026-09-01/nemotron-h-4b__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 20.86 | 1.97 | 12780.8 | 5841.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (conv/SSM recurrent state binds through the ExecutorMetadata section).
- Three-kind hybrid (24 Mamba2 + 24 MLP + 4 attention layers): only the 4 attention layers keep KV, so memory stays nearly flat with context length.
- On the composite 8-question probe one arithmetic near-miss appears identically on CPU and GPU — a quantization-level composite-prompt effect, not a backend bug (individual questions are 8/8 on both).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
