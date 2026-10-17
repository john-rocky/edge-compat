---
family: falcon-h1
license: other (falcon-llm-license)
model_id: falcon-h1r-7b-int8
source_url: https://huggingface.co/litert-community/Falcon-H1R-7B
task: text-generation
---

# falcon-h1r-7b-int8

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
| **Command** | `falcon_h1_work/convert_falcon_h1r_7b.sh (hf-to-litertlm): float export (6 prefill signatures 1024, 256, 64, 16, 4, 1 + decode, KV budget 4096, 176 state buffers = 44 KV pairs + 44 conv windows + 44 SSM states) -> dynamic int8 on every FULLY_CONNECTED + int8 embedding -> metadata (the bundle template, no start token, stop ids 11 / 228, the thought channel, fp32 GPU activations, max_num_tokens 4096)` |
| **Quantization** | int8 dynamic on every linear, int8 embedding; the Mamba conv and scan float; fp32 GPU activations declared (card L31, L106) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Falcon-H1R-7B_int8.litertlm` | `9bde28ab1e3026dfec0b16fb5d21e035e8fcb82b7e6c7cd9c679f898ef0a726e` | 7448.294 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.17.1/2026-10-14/falcon-h1r-7b-int8__mac-studio-m4-max.json`, `data/device_runs/0.18.0/2026-10-15/falcon-h1r-7b-int8__mac-studio-m4-max.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-studio-m4-max | cpu | pass | - | - | 1024 | - | 135.64 | 11.585 | 7638.3 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |
| mac-studio-m4-max | gpu | pass | - | - | - | - | - | - | - | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.17.1 · macOS 27.0 (26A428) | 2026-10-14 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 1024 | - | 565.05 | 35.875 | 1840.1 | - | Mac Studio (M4 Max, Mac16,9) · Apple M4 Max · litert-lm 0.18.0 · macOS 27.0 (26A428) | 2026-10-15 | measured |

## Pitfalls

- Reasoning model: it opens <think>…</think> on its own; the bundle declares the thought channel (<think> / </think>), so the runtime separates the reasoning from the answer and --thinking-budget works — with a 64-token budget the reasoning stopped mid-sentence and 17 × 23 was answered 481 (card L27).
- GPU precision: the file declares fp32 activations; with the runtime's default fp16 activations it answered none of the 8 check questions (<|pad|> from the first token to the cap), and an fp16-declared copy was only 1 % faster on the Mac GPU (9.17 vs 9.27 s) (card L63, L107).
- Galaxy S26: no row — in the int4 file's one GPU try (2026-10-15) the OpenCL delegate took every node of all 7 signatures and the phone rebooted 69 s into the warm-up; a device-lab Galaxy S26 of the same model line ran out of memory at the same point; the int8 file was not tried (after a sibling 8B int8 file dropped the same phone during its GPU warm-up we start no 8B-class int8 engine on it), nor the CPU (card L38-L40, L98).
- Pixel 8a: the int8 file was not tried — its 7.81 GB is twice the memory the phone had available at the start of the int4 try and almost its total RAM (card L99).
- iOS not measured: the file is one TFLite section of 7.27 GiB (the embedding table inside it), over the ~2 GiB single-section limit (card L43).
- Accuracy: GSM8K questions 0-99 with a 2,048-token budget, scored after </think> (an unclosed reply on its whole text): 96 / 100, 4 unclosed thoughts (3 right on their whole text); closed replies alone 93 of 96; the bf16 reference got 21 / 21 on questions 0-20, where this file got 20 (card L57-L63).
- Long output: on the Mac CPU, one long prompt (the refrigerator one) made this file reason for about 150 words, quote its own system message and then emit the six-character text `\u200b` to the 2,048-token cap, byte-identical in 3 of 3 runs; that prompt reaches the cap inside the thought on every form, the bf16 model included (card L51).
- Prompt-length check (17 lengths between 28 and 60 tokens): CPU 17/17 with the engine's prefill count equal to transformers', GPU 16/17 — at length 43 the thought quoted the prompt's filler words and repeated "well" to the 512-token cap once (card L49).
- One Engine per conversation: on litert-lm 0.18.0 a new Conversation on an Engine that already ran another starts from the earlier conversation's recurrent state (LiteRT-LM issue #3165); the pirate / one-word probe gave identical replies on a shared and a fresh Engine for this file, but the state is there, so give each conversation its own Engine (card L81).
- Chat template: TII's template rewritten for the runtime's engine (content parts as well as strings, past assistant turns rendered as the answer without the thought), equal to transformers' render of TII's template in 24 of 24 cells; TII's own template errors in 66 of 72 cells on the 0.18.0 engine; no start token; stop ids 11 / 228 plus the string \n<|im_end|> (card L49, L77).
- Tool calling: the model writes TII's <tool_call>{…}</tool_call> block as text after its thought; the bundle's generic model type has no tool-call parser, so the app parses it (card L79).
- Mac CPU: without --cache, `run` writes a compiled-weights cache next to the model (7.22 GB for this file); without a weight cache the CPU benchmark process reached 49 GB; on the Mac CPU this file is faster than the int4 file (int4's blockwise prefill runs at 44 % of int8's) (card L43, L75, L94).
- Runtime: tested on litert-lm 0.18.0 (Mac); on litert-lm 0.17.1 the file answered a plain `run` with a closed thought and then "Paris." (one GPU process) (card L23).

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
