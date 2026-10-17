---
family: nemotron-3-nano
license: other (nvidia-nemotron-open-model-license)
model_id: nemotron-3-nano-4b-int8
source_url: https://huggingface.co/litert-community/Nemotron-3-Nano-4B
task: text-generation
---

# nemotron-3-nano-4b-int8

| | |
|---|---|
| **Task** | text-generation |
| **Family** | nemotron-3-nano |
| **Source** | https://huggingface.co/litert-community/Nemotron-3-Nano-4B |
| **License** | other (nvidia-nemotron-open-model-license) |

## Conversion

| | |
|---|---|
| **Tool** | litert-torch plus a hybrid-cache patch, one command via hf-to-litertlm (python scripts/convert.py nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16; 1645 s on an M4 Max) (README Conversion notes) TODO (the README names the converter but not its version) |
| **Command** | `python scripts/convert.py nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (hf-to-litertlm)` |
| **Quantization** | int8 dynamic on linears + embedding (convs and the selective scan stay float, which keeps the hybrid state numerically sane); fp32 activations declared (README file table + Honest notes) |

## Artifacts

| File | SHA-256 | Size (MB) |
|---|---|---|
| `Nemotron-3-Nano-4B_int8.litertlm` | `e9274fb21c95944c5ad76b10a8294ebfe5b54a359d6d6166b1879a958331ca6a` | 3935.525 |

## Performance

No benchmark data yet.

## Device runs (NPU / LiteRT-LM)

Model-level on-device results — measured behavior of this exact artifact on the recorded device, only meaningful together with that environment. Throughput is conditional on the measured prompt length (Prompt (tokens); '-' = the source stated none): rows differing only there are different measurements, not re-runs. Full records (failure classes, log evidence): `data/device_runs/0.16.0/2026-08-27/nemotron-3-nano-4b-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.0/2026-09-05/nemotron-3-nano-4b-int8__galaxy-s26.json`, `data/device_runs/0.16.0/2026-10-13/nemotron-3-nano-4b-int8__mac-studio-m4-max.json`, `data/device_runs/0.16.1/2026-09-02/nemotron-3-nano-4b-int8__raspberry-pi-5.json`, `data/device_runs/0.18.0/2026-10-13/nemotron-3-nano-4b-int8__mac-studio-m4-max.json`, `data/device_runs/selfbuilt-b2f686e2/2026-10-13/nemotron-3-nano-4b-int8__galaxy-s26.json`.

| Device | Accelerator | Status | Full delegation | Output match | Prompt (tokens) | Latency p50 (ms) | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak mem (MB) | Environment | Date | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-s26 | cpu | fallback | no | - | sig:three-turn conversation, thinking default | - | - | - | - | - | Galaxy S26 (SM-S942Q) · SM8850 · litert-lm selfbuilt-b2f686e2 · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | cpu | fallback | no | - | sig:three-turn conversation, thinking off | - | - | - | - | - | Galaxy S26 (SM-S942Q) · SM8850 · litert-lm selfbuilt-b2f686e2 · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | cpu | fallback | no | - | sig:three-turn conversation, thinking on | - | - | - | - | - | Galaxy S26 (SM-S942Q) · SM8850 · litert-lm selfbuilt-b2f686e2 · Android 16 (BP4A.251205.006) | 2026-10-13 | measured |
| galaxy-s26 | cpu | fallback | no | - | 211 | - | 52.94 | 12.87 | 4060.0 | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| galaxy-s26 | gpu | run_failed | yes | - | - | - | - | - | - | - | Galaxy S26 (SM-S942Q) · Qualcomm SM8850 · litert-lm 0.16.0 · Android 16 | 2026-09-05 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:8-question gate (CLI run --cache no) | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:one prompt (CLI run --cache no) | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.16.0 | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:three-turn conversation, thinking off (Python API) | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | sig:three-turn conversation, thinking unset (Python API) | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| mac-studio-m4-max | cpu | pass | - | - | 256 | - | 99.57 | 22.73 | 2643.2 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| mac-studio-m4-max | gpu | pass | yes | - | sig:8-question gate (CLI run --backend gpu --cache no) | - | - | - | - | - | Mac Studio (M4 Max) · litert-lm 0.18.0 | 2026-10-13 | measured |
| mac-studio-m4-max | gpu | pass | - | - | 256 | - | 803.37 | 83.3 | 330.7 | - | mac-studio-m4-max · Apple M4 Max · litert-lm 0.16.0 | 2026-08-27 | measured |
| raspberry-pi-5 | cpu | pass | - | - | 256 | - | 22.72 | 2.2 | 11721.5 | 5167.0 | Raspberry Pi 5 Model B Rev 1.1 · Broadcom BCM2712 · litert-lm 0.16.1 · Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41 | 2026-09-02 | measured |

## Pitfalls

- Requires litert-lm >= 0.15 (README intro).
- GPU requires --cache no on this bundle: with the compiled-graph cache enabled, litert-lm run --backend gpu fails with WebGPU 'Invalid BindGroup' validation errors and the 8-question sweep returns token soup (0/8); the same file with --cache no answers 8/8 (README Honest notes).
- 8-question gate 7/8 on CPU, 8/8 on GPU (litert-lm 0.16.0, M4 Max); the single CPU miss is the rhyme item ('violets are purple'); chat template byte-equal to the source (10,504/10,504 bytes — the source tokenizer_config.json carries a different 10,497-byte copy); stops <|im_end|> (id 11) + id 2; no spurious start token (README Correctness).
- Reasoning model — answers arrive after a <think> block, so give it a token budget that fits the thought (the gate used 3200); >=3B exports use a reduced 7-signature prefill ladder because every signature costs engine RAM (README Honest + Conversion notes).
- Phones: one Galaxy S26, CPU only — the CPU backend runs this file (litert_lm_advanced_main v0.16.0, 2026-09-05), the GPU backend crashes with SIGSEGV (exit 139) while the OpenCL delegate initializes its kernels for prefill_1024 after accepting all 2,827 ops; no Mali phone has run it and no 8 GB phone has been tried (the sibling Nemotron-H-4B did not fit one: an expectation, not a measurement of this file) (README Honest notes).
- Since 2026-10-13 (file sha256 e9274fb2…, metadata only, weights unchanged) the chat template renders every assistant history turn as <think></think> plus its answer; the earlier file (feefdcd5…) kept the think block only on the latest assistant turn, so on litert-lm 0.18.0 the second message of a conversation with thinking on failed with INTERNAL: The new rendered template string does not start with the previous rendered template string (LiteRT-LM#3166) — re-download a file fetched before 2026-10-13. One Engine per conversation still (a second Conversation on the same Engine starts from the earlier recurrent state, LiteRT-LM#3165). On the litert-lm 0.18.0 CLI the GPU path still needs --cache no (README Honest notes + 2026-10-13 dated note).
- auto_map in the config is not proof of remote code: transformers registers nemotron_h natively, so the library implementation loads without trust_remote_code (README Conversion notes).
- LiteRT-LM .litertlm bundle: LiteRT.js cannot run it, so delegation stays null and there is no browser block; device rows come from data/device_runs/.

---

_Rendered by `edge-card` from `card.json` (the source of truth); edit the inputs, not this file._
