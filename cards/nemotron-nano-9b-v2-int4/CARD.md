---
family: nemotron-h
license: other (nvidia-open-model-license)
model_id: nemotron-nano-9b-v2-int4
source_url: https://huggingface.co/litert-community/NVIDIA-Nemotron-Nano-9B-v2
task: text-generation
---

# nemotron-nano-9b-v2-int4

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nemotron-h |
| **Source** | https://huggingface.co/litert-community/NVIDIA-Nemotron-Nano-9B-v2 |
| **License** | other (nvidia-open-model-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch (commit 115a136 + the Nemotron-H patch of hf-to-litertlm nemotron_h_work/nemotron_h_litert_torch.patch + a second patch for NVIDIA's dt arithmetic; lightweight export, 1e8-byte lazy-constant threshold); repro = hf-to-litertlm nemotron_h_work/convert_nemotron_nano_9b.sh litert-torch 115a136 + Nemotron-H patch + dt patch (torch 2.12.1, transformers 5.14.1, ai-edge-quantizer 0.8.0, litert-converter 0.3.0); packed and checked with litert-lm 0.18.0 |
| **Command** | `nemotron_h_work/convert_nemotron_nano_9b.sh (hf-to-litertlm): nvidia/NVIDIA-Nemotron-Nano-9B-v2 at 6533e8de -> float export (6 prefill signatures 1024, 256, 64, 16, 4, 1 + decode, KV budget 4096, 62 state buffers = 4 KV pairs + 27 conv windows + 27 SSM states) -> int4 blockwise-32 (min-max) on the linear layers with every block scale floored at 2^-14 before quantizing, int8 lm_head and embedding -> metadata (the bundle template, no start token, stop ids 2 / 11 / 12, the thought channel, fp32 GPU activations, max_num_tokens 4096)` |
| **Quantization** | int4 blockwise-32 (min-max) on the linear layers, block scales floored at 2^-14 (the plain build left 66,388 zero or subnormal fp16 block scales of 240,988,160, which XNNPACK refuses), int8 lm_head and embedding; fp32 GPU activations declared (card L41, L118-L119) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `NVIDIA-Nemotron-Nano-9B-v2_int4.litertlm` | `9b0bd48b3695051274cd738c4a7931198e2f830fc268bca59573eab0a07d84a3` | 5333.509 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.18.0/2026-10-15/nemotron-nano-9b-v2-int4__mac-studio-m4-max.json`, `data/device_runs/prebuilt-adac974b/2026-10-15/nemotron-nano-9b-v2-int4__pixel-8a.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 56.865 | 13.985 | 18084.6 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 478.265 | 45.52 | 2163.5 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| pixel-8a | gpu | load_failed | - | - | - | - | - | - | - | - | Pixel 8a · Tensor G3 · litert-lm prebuilt-adac974b · Android 16 (CP1A.260505.005) | 2026-10-15 | measured |

## Pitfalls

- Reasoning model: it opens a <think> block unless told otherwise; the bundle declares the thought channel (<think>\n / \n</think>), so the runtime returns the reasoning separately. NVIDIA's switch is text: /no_think in the system message or a user message turns the reasoning off (the template then ends the prompt with <think></think>), /think turns it back on; the CLI's --thinking false leaves the output byte for byte the same (the template reads no enable_thinking variable) (card L36, L88).
- Block scales: XNNPACK prepares a blockwise int4 layer only when every scale is a normal float; the plain int4 build left 66,388 zero or subnormal fp16 block scales, so this file floors them at 2^-14 before quantizing (card L118).
- GPU precision: the file declares fp32 activations; with the runtime's default fp16 activations the int8 weights answered none of the 8 check questions (every reply <unk> repeated to the cap) (card L56, L73, L119).
- Accuracy reference: GSM8K n=100 (thinking off, greedy, Mac GPU) int4 92 / int8 93 against transformers on MPS 83 (bf16) / 84 (fp32); the whole gap is on the 16 prompts longer than one 128-token Mamba2 chunk, where transformers' torch scan (no CUDA kernels) splits the prefill: re-run with the whole prompt in one chunk the reference scored the files' 15 of 16 (card L43, L64-L73).
- dt arithmetic: transformers 5.14.1 clamps each Mamba2 dt at 0.001 from below, NVIDIA's own modeling code does not; the files follow NVIDIA's code (339 of 3,456 heads rest below 0.001) (card L116).
- Tool calling: the template renders NVIDIA's tool form, and the model writes <TOOLCALL>[{"name": ..., "arguments": ...}]</TOOLCALL> as text; the generic model type has no parser for it, so an app parses the calls itself (card L90).
- One Engine per conversation: litert-lm 0.18.0 starts a new Conversation on a used Engine from the earlier conversation's recurrent state (LiteRT-LM issue 3165); the 27 Mamba2 states make this file exposed to it (card L92).
- Galaxy S26: not run — a sibling Mamba2-hybrid file of the same size class (Falcon-H1R-7B int4, 4.69 GB, the same binary) took the phone off USB 69 s into its GPU engine warm-up on 2026-10-15, and one such drop ends the tries on that phone; the CPU multipliers put the file above the phone's memory (card L109).
- Pixel 8a: the one GPU try was stopped by the harness's memory rule at MemAvailable 382,932 kB, 41 s after the start, after prefill_1024 was delegated 3582/3582 to the OpenCL delegate and while the runtime prepared the weights on the CPU side (Mali); no restart; GPU execution on a Mali phone is not verified (card L110).
- Shape: a three-signature variant (prefill 1024 / 256 / 1, the same weights) took the same time on the 1024 / 256 workload with a faster engine init (9.0 vs 13.8 s) and a slower first token on a 15-token prompt; the six-signature file ships (card L105).
- iOS not measured: the file is one 5.21 GiB TFLite section, over the ~2 GiB single-section limit (card L52).
- Mac: the int4 file is the faster file on the GPU (7.77 vs 8.37 s, decode 45.5 vs 41.0 tok/s) and the slower one on the CPU (its blockwise prefill 56.9 vs 127.9 tok/s); without --cache, the first CPU `run` writes a 4.94 GB compiled-weights cache next to the model (card L52, L86).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
