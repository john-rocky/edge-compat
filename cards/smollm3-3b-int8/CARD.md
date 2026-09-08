---
family: smollm3
license: apache-2.0
model_id: smollm3-3b-int8
source_url: https://huggingface.co/litert-community/SmolLM3-3B
task: text-generation
---

# smollm3-3b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | smollm3 |
| **Source** | https://huggingface.co/litert-community/SmolLM3-3B |
| **License** | apache-2.0 |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch export_hf (README Conversion, written for the repo's q4 file; SmolLM3ForCausalLM on the existing converter, NoPE attention lowers to generic ops) TODO (the README names the converter but not its version) |
| **Command** | `TODO (not stated in the README)` |
| **Quantization** | TODO — the README describes this ~3.1 GB file only as 'the repo also carries SmolLM3-3B.litertlm, gated on the Galaxy S26 GPU'; the id suffix -int8 is the device-run lane's, not a README statement |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `SmolLM3-3B.litertlm` | `f9aa844c27f5ccfae9cb701c7ca63f2faba501ed187aacc85044b6200df15a37` | 2965.286 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-24/smollm3-3b-int8__galaxy-s26.json`, `data/device_runs/0.16.1/2026-09-01/smollm3-3b-int8__raspberry-pi-5.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | gpu | pass | yes | - | 211 | - | 312.27 | 14.41 | 750.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-08-24 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 19.1 | 1.97 | 14036.7 | 4247.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-01 | measured |

## Pitfalls

- Galaxy S26 GPU per the README: runs, 2930/2930 ops across 2 subgraphs on LiteRT GPU (704 MB); Raspberry Pi 5 CPU 19.1 tok/s prefill / 2.0 decode (README tables); this repo's Pi 5 row is the wave-2 journal (device runs).
- The tokenizer in SmolLM3-3B.litertlm was a SentencePiece conversion of the model's BPE tokenizer that lost byte-level semantics for standalone accented letters/symbols, and both bundles later received a metadata-only repair declaring the reasoning channel (thought, <think>…</think>) and a default-system-turn change (every section except LlmMetadata byte-identical, sha256 per section) — read the README's 'Metadata-only change' notes for the exact state of the file you download (README).
- License Apache-2.0, inherited from HuggingFaceTB/SmolLM3-3B (README License).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
