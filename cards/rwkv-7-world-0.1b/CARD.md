---
family: rwkv7
license: apache-2.0
model_id: rwkv-7-world-0.1b
source_url: https://huggingface.co/litert-community/RWKV-7-World-0.1B-LiteRT
task: text-generation
---

# rwkv-7-world-0.1b

| | |
|---|---|
| **Task** | text-generation |
| **Family** | rwkv7 |
| **Source** | https://huggingface.co/litert-community/RWKV-7-World-0.1B-LiteRT |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch 0.10.0 (editable dev checkout 115a136 + local patches) |
| **Command** | `python rwkv7/scripts/build_rwkv7_step.py` |
| **Quantization** | fp16 |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `b3ea0ffc92e6b3154ac228c7e98e0ba58368e3b41ee648d21ac9ecaa0426be97.tflite` | `583fc0bad83fb8252925c305eecdbdeff1bc02ebbd7cc071c0393f9182486728` | 268.966 |

## Performance

No benchmark data yet.

## Browser (LiteRT.js)

Model-level sweep results — measured behavior of this exact `.tflite` in the browser, only meaningful together with the environment that produced it. Full records (failure class, error evidence, sweep config): `data/sweep/2.5.3/2026-08-11/rwkv-7-world-0.1b.json`.

| Backend | Status | Full delegation | Output match | Max rel diff | Latency p50 (ms) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|
| wasm_xnnpack | pass | - | - | - | 22.328 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |
| webgpu_mldrift | pass | yes | pass | 0.008479067302596715 | 15.07 | mac-studio-m4-max · chromium 151.0.7922.34 headless · macOS · @litertjs/core 2.5.3 | 2026-08-11 | measured |

## Pitfalls

- Host side per step: token-embedding row lookup from the ~100 MB fp16 table (GATHER is GPU-banned), argmax over the 65536 logits, and recycling the three recurrent states into the next step (zoo README L1097).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
