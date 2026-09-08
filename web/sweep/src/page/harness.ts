/**
 * In-page sweep harness, bundled by esbuild and served to headless Chromium.
 *
 * Holds at most one compiled model at a time. Tensors are manually managed in
 * LiteRT.js: every tensor created here is deleted before the function returns,
 * so leaks cannot skew latency across runs (spec 5.2).
 */
import {
  type CompiledModel,
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';

import type {
  FixtureInput,
  InitResult,
  LoadResult,
  PageAccelerator,
  ProbeResult,
  RunResult,
  SweepPageApi,
} from '../shared/protocol.ts';

let compiledModel: CompiledModel | null = null;

function errorText(err: unknown): string {
  if (err instanceof Error) {
    return `${err.name}: ${err.message}`;
  }
  return String(err);
}

async function init(wasmBaseUrl: string, jspi: boolean): Promise<InitResult> {
  try {
    await loadLiteRt(wasmBaseUrl, { jspi });
    return { ok: true, jspi, error: null };
  } catch (err) {
    if (jspi) {
      // JSPI variant failed to load; retry plain so the sweep can still
      // measure, recording jspi: false honestly in env.
      try {
        await loadLiteRt(wasmBaseUrl, { jspi: false });
        return { ok: true, jspi: false, error: null };
      } catch (retryErr) {
        return { ok: false, jspi: false, error: errorText(retryErr) };
      }
    }
    return { ok: false, jspi: false, error: errorText(err) };
  }
}

async function probe(): Promise<ProbeResult> {
  const webgpuSupported = isWebGPUSupported();
  let adapter: ProbeResult['adapter'] = null;
  if (webgpuSupported) {
    try {
      const gpuAdapter = await navigator.gpu.requestAdapter();
      if (gpuAdapter !== null) {
        const info = gpuAdapter.info;
        adapter = {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
        };
      }
    } catch {
      adapter = null;
    }
  }
  return { webgpuSupported, adapter };
}

async function loadModel(url: string, accelerator: PageAccelerator): Promise<LoadResult> {
  if (compiledModel !== null) {
    compiledModel.delete();
    compiledModel = null;
  }
  try {
    compiledModel = await loadAndCompile(url, { accelerator });
    return { ok: true, fullyAccelerated: compiledModel.isFullyAccelerated, error: null };
  } catch (err) {
    compiledModel = null;
    return { ok: false, fullyAccelerated: null, error: errorText(err) };
  }
}

function makeInputTensors(inputs: FixtureInput[]): Tensor[] {
  const tensors: Tensor[] = [];
  try {
    for (const input of inputs) {
      const data =
        input.dtype === 'int32' ? new Int32Array(input.data) : new Float32Array(input.data);
      tensors.push(Tensor.fromTypedArray(data, input.shape));
    }
    return tensors;
  } catch (err) {
    for (const tensor of tensors) {
      tensor.delete();
    }
    throw err;
  }
}

function deleteAll(tensors: readonly Tensor[]): void {
  for (const tensor of tensors) {
    if (!tensor.deleted) {
      tensor.delete();
    }
  }
}

async function runModel(
  inputs: FixtureInput[],
  warmupRuns: number,
  timedRuns: number,
): Promise<RunResult> {
  if (compiledModel === null) {
    return { ok: false, latenciesMs: [], outputs: [], error: 'no model loaded' };
  }
  const model = compiledModel;
  let inputTensors: Tensor[] = [];
  try {
    inputTensors = makeInputTensors(inputs);

    for (let i = 0; i < warmupRuns; i++) {
      const outputs = await model.run(inputTensors);
      // run() reports some backend failures (e.g. "failed to create XNNPACK
      // runtime") only on the console and resolves with zero outputs — that
      // is a failed run, not a 0.3 ms success.
      if (outputs.length === 0) {
        return {
          ok: false,
          latenciesMs: [],
          outputs: [],
          error: 'model.run returned no outputs (backend runtime creation likely failed; see delegation evidence)',
        };
      }
      // Read one value to force completion before deleting.
      await outputs[0]!.data();
      deleteAll(outputs);
    }

    const latenciesMs: number[] = [];
    let finalOutputs: number[][] = [];
    for (let i = 0; i < timedRuns; i++) {
      const start = performance.now();
      const outputs = await model.run(inputTensors);
      if (outputs.length === 0) {
        return {
          ok: false,
          latenciesMs: [],
          outputs: [],
          error: 'model.run returned no outputs (backend runtime creation likely failed; see delegation evidence)',
        };
      }
      const outputData: number[][] = [];
      for (const output of outputs) {
        outputData.push(Array.from(await output.data(), Number));
      }
      latenciesMs.push(performance.now() - start);
      deleteAll(outputs);
      finalOutputs = outputData;
    }
    return { ok: true, latenciesMs, outputs: finalOutputs, error: null };
  } catch (err) {
    return { ok: false, latenciesMs: [], outputs: [], error: errorText(err) };
  } finally {
    deleteAll(inputTensors);
  }
}

const api: SweepPageApi = { init, probe, loadModel, runModel };
window.litertSweep = api;
