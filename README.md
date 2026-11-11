# edge-compat

**Will this `.tflite` run on the GPU, the NPU or in the browser, and where will it fall back?** edge-compat answers that for [LiteRT](https://ai.google.dev/edge/litert), Google's on-device runtime and the new name for TensorFlow Lite. I ran models on phones, browsers and a Raspberry Pi and recorded what each op did, so you can check your own model before you test it on hardware.

Built by john-rocky. Measurements and views are my own.
edge-compat is an unofficial project for LiteRT, not an official Google product.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/john-rocky/edge-compat/main/docs/images/model-status-dark.svg">
  <img alt="Which models ran where. Browser, WebGPU: 69 loaded and ran with no fallback or mismatch recorded, 61 fell back, gave wrong output, or failed to load or run. Browser, WASM: 108 loaded and ran with no fallback or mismatch recorded, 22 fell back, gave wrong output, or failed to load or run. Galaxy S26, GPU: 56 loaded and ran with no fallback or mismatch recorded, 3 fell back, gave wrong output, or failed to load or run. Galaxy S26, NPU: 55 loaded and ran with no fallback or mismatch recorded, 7 fell back, gave wrong output, or failed to load or run. Raspberry Pi 5, CPU: 87 loaded and ran with no fallback or mismatch recorded, 0 fell back, gave wrong output, or failed to load or run." src="https://raw.githubusercontent.com/john-rocky/edge-compat/main/docs/images/model-status-light.svg" width="860">
</picture>
</p>

A pass means the file loaded and ran, and no fallback or mismatch was recorded. Output was compared for 65 of the 69 WebGPU passes, against the WASM result of the same file. WASM is that reference, and the Galaxy S26 and Raspberry Pi passes come from speed runs, so their output was not compared.

<a name="edge-lint--static-delegate-pre-flight"></a>

## Try it

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). `edge-lint` reads your model without running it, so you need no LiteRT runtime and no device. Its `--matrix` file is my record of what each op does on one backend at one runtime version. This one is for Android GPUs:

```sh
curl -sO https://john-rocky.github.io/edge-compat/data/matrix/gpu_mldrift__2.2.0.json
uvx --from edge-compat edge-lint model.tflite --matrix gpu_mldrift__2.2.0.json
```

For another target, swap the file. The number is the runtime version, and [`data/matrix/`](data/matrix/) has them all:

| Target | Matrix file | Entries |
|---|---|---:|
| Android GPU (ML Drift, LiteRT's GPU delegate) | `gpu_mldrift__2.2.0.json` | 28 |
| Qualcomm NPU | `npu_qnn_htp_android__2.2.0.json` | 4 |
| Browser, WebGPU | `webgpu_mldrift__2.5.3.json` | 132 |
| Browser, WASM | `wasm_xnnpack__2.5.3.json` | 127 |
| CPU | `cpu_xnnpack__2.1.6.json` | 153 |
| iPhone GPU (Metal) | `gpu_metal_ios__0.15.0.json` | 4 |

An op with no entry reports `unknown`. A report holds only for that runtime version, because support changes between versions.

If you don't have a model yet, try the example files. They are made up for this demo:

```sh
B=https://raw.githubusercontent.com/john-rocky/edge-compat/main/data/examples
curl -sO $B/model_mixed_example.tflite
curl -sO $B/matrix_example.json
uvx --from edge-compat edge-lint model_mixed_example.tflite --matrix matrix_example.json
```

The report starts like this:

```text
gpu_mldrift @ litert 0.0.0-example — 40.0% of 5 ops delegated · 1 partition · 3 blocking ops
model:  model_mixed_example.tflite (sha256 9af4f555f718…, 1 subgraph(s))
matrix: matrix_example.json (generated 2026-08-10)

Non-delegated ops:
  s0/n0  FULLY_CONNECTED  dtypes=int8  fallback  provenance=example
      reason: matched entry specific to op, dtypes, and constraints
  s0/n1  CUSTOM  dtypes=float32  unknown  provenance=-
      reason: custom op 'ExampleCustomOp' — not a builtin, outside the matrix vocabulary
  s0/n4  TRANSPOSE_CONV  dtypes=float32  fallback  provenance=example
      reason: matched entry specific to op, dtypes, and constraints
      condition: rejected by some mobile GPU drivers regardless of rank
```

After that it shows where the model splits between the CPU and the GPU, and the rewrite recorded for each op.

To browse instead, [the site](https://john-rocky.github.io/edge-compat/) lists each model with its status per backend. Its [Check your model](https://john-rocky.github.io/edge-compat/check/) page runs your `.tflite` in your own browser without uploading it, so its answer is about that browser and not a phone.

## What you can look up

| To find | Look in |
|---|---|
| Which ops a backend accepts, per runtime version | [`data/matrix/`](data/matrix/) |
| Why a model falls back or returns wrong numbers | The `edge-lint` report |
| Whether a model runs in the browser, and how fast | [The site](https://john-rocky.github.io/edge-compat/) and [`data/sweep/`](data/sweep/) |
| How fast a model ran on a device | Its card, from [the card list](cards/README.md) |

## Where the data comes from

Each entry in a matrix file records its source: `measured`, `vendor_doc`, `inferred` or `example`. When I copy a matrix to a new runtime version, its `measured` entries become `inferred` until I measure them again.

An op gets a `measured` entry only when a runtime log names it or a single-op test ran it. A whole model passing or failing does not count.

## Tools

| Command | What it does |
|---|---|
| `edge-lint` | Predicts fallback for a `.tflite` or `.litertlm` file, as text, Markdown or JSON, with exit codes for CI |
| `edge-fix` | Applies a recorded rewrite to a copy of your model, then checks that more of it runs on the target and the output still matches. Dry run by default. |
| `edge-card` | Builds a model card from measured results |
| `edge-compat` | Checks and updates the matrix files |
| MCP server | Gives AI agents the same data over stdio, see [`mcp/README.md`](mcp/README.md) |

To work from a clone:

```sh
uv sync            # or: pip install -e .
uv run edge-compat --help
```

## Documentation

- [How the project works](docs/how-it-works.md): the data rules and how this differs from similar tools
- [Matrix commands](docs/cli.md): `edge-compat matrix`, `probe` and the optional runners
- [edge-lint](docs/edge-lint.md): the checks, the report and a CI example
- [edge-card](docs/edge-card.md): the card format and its commands
- [edge-fix](docs/edge-fix.md): how a model gets rewritten and verified
- [Browser](docs/browser.md): the sweep harness, the site and the web backends
- [Device runs](docs/device-runs.md): NPU and LLM results, and the regression suite
- [Release pipeline](docs/release-pipeline.md): what happens when a new runtime ships
- [Freshness](docs/freshness.md): how the data is kept current
- [Python API and CSV format](docs/matrix-api.md)
- [Repository layout and development](docs/development.md)

## Contributing

I develop this in a private repository and export it here, one commit per export ([`EXPORT.md`](EXPORT.md)). Open issues here. A fix goes into the private repository and comes out with the next export.

## License

Apache-2.0 (see `LICENSE`).
