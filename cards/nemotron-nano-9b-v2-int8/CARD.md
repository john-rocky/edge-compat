---
family: nemotron-h
license: other (nvidia-open-model-license)
model_id: nemotron-nano-9b-v2-int8
source_url: https://huggingface.co/litert-community/NVIDIA-Nemotron-Nano-9B-v2
task: text-generation
---

# nemotron-nano-9b-v2-int8

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
| **Command** | `nemotron_h_work/convert_nemotron_nano_9b.sh (hf-to-litertlm): nvidia/NVIDIA-Nemotron-Nano-9B-v2 at 6533e8de -> float export (6 prefill signatures 1024, 256, 64, 16, 4, 1 + decode, KV budget 4096, 62 state buffers = 4 KV pairs + 27 conv windows + 27 SSM states) -> dynamic int8 on every FULLY_CONNECTED + int8 embedding -> metadata (the bundle template, no start token, stop ids 2 / 11 / 12, the thought channel, fp32 GPU activations, max_num_tokens 4096)` |
| **Quantization** | int8 dynamic on every linear layer, int8 embedding; the Mamba conv and the selective scan float; fp32 GPU activations declared (card L40, L118-L119) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `NVIDIA-Nemotron-Nano-9B-v2_int8.litertlm` | `575505ddb315537b347ad1b5bc3385f5770f4c0152840884ae825e394ed3a194` | 8651.981 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/nemotron-nano-9b-v2-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/nemotron-nano-9b-v2-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 127.93 | 12.39 | 8085.8 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 480.175 | 41.02 | 2157.0 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Reasoning model: it opens a <think> block unless told otherwise; the bundle declares the thought channel (<think>\n / \n</think>), so the runtime returns the reasoning separately. NVIDIA's switch is text: /no_think in the system message or a user message turns the reasoning off (the template then ends the prompt with <think></think>), /think turns it back on; the CLI's --thinking false leaves the output byte for byte the same (the template reads no enable_thinking variable) (card L36, L88).
- GPU precision: the file declares fp32 activations; with the runtime's default fp16 activations it answered none of the 8 check questions (every reply <unk> repeated to the cap), for 1.5 % less time (card L56, L73, L119).
- Accuracy reference: GSM8K n=100 (thinking off, greedy, Mac GPU) int8 93 / int4 92 against transformers on MPS 83 (bf16) / 84 (fp32); the whole gap is on the 16 prompts longer than one 128-token Mamba2 chunk, where transformers' torch scan (no CUDA kernels) splits the prefill: re-run with the whole prompt in one chunk the reference scored the files' 15 of 16 (card L43, L64-L73).
- dt arithmetic: transformers 5.14.1 clamps each Mamba2 dt at 0.001 from below, NVIDIA's own modeling code does not; the files follow NVIDIA's code (339 of 3,456 heads rest below 0.001); an int8 build with the transformers clamp scored the same 93 on GSM8K and differs by one MAXIMUM op per mixer (card L116).
- Tool calling: the template renders NVIDIA's tool form, and the file writes <TOOLCALL>[{"name": ..., "arguments": ...}]</TOOLCALL> as text; the generic model type has no parser for it, so an app parses the calls itself (card L90).
- One Engine per conversation: litert-lm 0.18.0 starts a new Conversation on a used Engine from the earlier conversation's recurrent state (LiteRT-LM issue 3165); the probe on this file gave the same answers on a shared and a fresh Engine, but the 27 Mamba2 states are there all the same (card L92).
- Chat template: NVIDIA's template does not parse on the 0.18.0 runtime (its rewrite of .split('</think>')[1] is a syntax error); the bundle's template is NVIDIA's with eleven edits (content parts, the string methods as filters, the trailing-assistant special case removed), equal to transformers' render in 39 of 39 cells; no start token; stop ids 2 / 11 / 12; the tokenizer section is the repo's tokenizer.json byte for byte (card L88).
- Galaxy S26: not run — a sibling Mamba2-hybrid file of the same size class (Falcon-H1R-7B int4, 4.69 GB) took the phone off USB 69 s into its GPU engine warm-up on 2026-10-15, and no 8B-class int8 engine is started on that phone after an 8B int8 file restarted it during its GPU warm-up; the CPU multipliers put both files above the phone's memory (card L109).
- Pixel 8a: the int8 file was not tried (9.07 GB, larger than the phone's RAM); NPU not run (no HTP path for dynamic-range int8) (card L110-L111).
- iOS not measured: the file is one 8.45 GiB TFLite section, over the ~2 GiB single-section limit (card L52).
- Mac CPU: without --cache, `run` writes a compiled-weights cache next to the model on its first CPU run (8.32 GB for this file); the int8 file is the faster file on the Mac CPU (28.68 vs 36.32 s), the int4 file on the GPU (card L52, L86).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
