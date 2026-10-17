---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1r-7b-int4
source_url: https://huggingface.co/litert-community/Falcon-H1R-7B
task: text-generation
---

# falcon-h1r-7b-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | falcon-h1 |
| **Source** | https://huggingface.co/litert-community/Falcon-H1R-7B |
| **License** | other (falcon-llm-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (commit 115a136 + the Falcon-H1 patch of hf-to-litertlm falcon_h1_work/falcon_h1_litert_torch.patch; lightweight export, 1e8-byte lazy-constant threshold); repro = hf-to-litertlm falcon_h1_work/convert_falcon_h1r_7b.sh litert-torch 115a136 + Falcon-H1 patch (torch 2.12.1, transformers 5.14.1, ai-edge-quantizer 0.8.0, litert-converter 0.3.0); packed by the litert-lm 0.18.0 CLI |
| **Command** | `falcon_h1_work/convert_falcon_h1r_7b.sh (hf-to-litertlm): float export (6 prefill signatures 1024, 256, 64, 16, 4, 1 + decode, KV budget 4096, 176 state buffers = 44 KV pairs + 44 conv windows + 44 SSM states) -> int4 blockwise-32 min-max on the linears with the block scales floored at 2^-14, int8 output projection and embedding -> metadata (the bundle template, no start token, stop ids 11 / 228, the thought channel, fp32 GPU activations, max_num_tokens 4096)` |
| **Quantization** | int4 blockwise-32 on the linears (block scales floored at 2^-14), int8 output projection and embedding; fp32 GPU activations declared (card L32, L106) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1R-7B_int4.litertlm` | `7a8820ae4f38bc82b40b06f181ed013c012ec38dbf5329b866d68183080a8ea1` | 4471.815 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-15/falcon-h1r-7b-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/falcon-h1r-7b-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 60.325 | 12.35 | 17065.6 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 565.775 | 39.705 | 1835.1 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Google Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-15 | measured |

## Pitfalls

- Block scales: the plain int4 build left 96 zero and 748 subnormal fp16 block scales (dead 32-weight blocks in 7 Mamba-mixer tensors), and XNNPACK prepares a blockwise int4 layer only when every scale is a normal float; this file floors its block scales at 2^-14 (847 of 212,044,800 scales; the weights are otherwise the plain build's) (card L106).
- Reasoning model: it opens <think>…</think> on its own; the bundle declares the thought channel (<think> / </think>), so the runtime separates the reasoning from the answer and --thinking-budget works (card L27).
- GPU precision: the file declares fp32 activations; with the runtime's default fp16 activations the int8 file of this conversion answered none of the 8 check questions (<|pad|> from the first token) (card L63, L107).
- Galaxy S26: no row — in the one GPU try (2026-10-15) the OpenCL delegate took every node of all 7 signatures and 69 s after the start, during the warm-up, the phone rebooted (its uptime dates the boot 17 s after it left USB) and stayed locked until its owner unlocked it; not run again, CPU not tried; on a device-lab Galaxy S26 of the same model line (SM-S942U1) the low-memory killer removed 92 apps in 27 seconds and the system services died 11 s after the last runtime line — the GPU engine of this file does not build on a 12 GB Galaxy S26 (card L38, L98).
- Pixel 8a: the one GPU try was stopped by the harness's memory rule (MemAvailable 307,456 kB 50 s after the start, from 3,888,840 kB, after prefill_1024 was delegated 6865/6865 to the OpenCL delegate; the runtime prepares weights on the CPU side for Mali GPUs; no restart) (card L99).
- iOS not measured: the file is one TFLite section of 4.36 GiB (the embedding table inside it), over the ~2 GiB single-section limit (card L43).
- Accuracy: GSM8K questions 0-99 with a 2,048-token budget, scored after </think> (an unclosed reply on its whole text): 98 / 100, 10 unclosed thoughts (9 right on their whole text); closed replies alone 89 of 90; counting every unclosed reply as wrong 89; the bf16 reference got 21 / 21 on questions 0-20, as did this file (card L57-L63).
- Prompt-length check (17 lengths between 28 and 60 tokens): GPU 17/17, CPU 16/17 — at length 41 the thought quoted the prompt's filler words and repeated "well" to the 512-token cap in 3 of 3 runs (card L49).
- Answers: this file's replies use narrow no-break spaces (U+202F) inside dates ("May 3 and May 17."), so a byte comparison with a typed string fails (card L75).
- One Engine per conversation: on litert-lm 0.18.0 a new Conversation on an Engine that already ran another starts from the earlier conversation's recurrent state (LiteRT-LM issue #3165); give each conversation its own Engine (card L81).
- Mac: the faster file on the GPU (8.26 vs 8.95 s for the 1024 / 256 workload, decode 39.7 vs 35.9 tok/s), the slower on the CPU (its blockwise prefill runs at 44 % of int8's); without --cache, `run` writes a 4.24 GB compiled-weights cache next to the model; a three-signature variant (prefill 1024 / 256 / 1) took 2.5 % less time with a shorter init and a slower first token on a 15-token prompt — the six-signature file ships (card L36-L43, L75, L94).
- Runtime: tested on litert-lm 0.18.0 (Mac); the int4 file was not run on litert-lm 0.17.1 (card L23).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
