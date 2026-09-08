/**
 * Per-model demo page runtime, bundled by esbuild into assets/demo.js.
 *
 * Vanilla TypeScript + @litertjs/core, no framework. The model is fetched at
 * runtime from its official source URL (weights are never part of the site);
 * sample inputs are generated in-page with the same seeded PRNG the sweep
 * used. Tensors are manually managed: everything created here is deleted
 * before the function returns, mirroring the sweep harness.
 */
import {
  type CompiledModel,
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';

import type { DemoConfig } from '../shared/config.ts';
import { type SampleInput, buildSampleInputs, p50 } from './prng.ts';

type Accelerator = 'wasm' | 'webgpu';

/**
 * Measurement hook: page loads, backend used, CTA copies. Deliberately a
 * no-op — the counter choice is the owner's (DECISIONS #69: repo-stats only
 * until an analytics stance is picked). No cookies, no PII, ever.
 */
function measure(_event: string): void {
  // intentionally empty
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (el === null) {
    throw new Error(`missing element #${id}`);
  }
  return el as T;
}

function readConfig(): DemoConfig {
  const raw = byId<HTMLScriptElement>('demo-config').textContent ?? '';
  return JSON.parse(raw) as DemoConfig;
}

function errorText(err: unknown): string {
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err);
}

let litertReady: Promise<boolean> | null = null;
let jspiActive = false;
let compiledModel: CompiledModel | null = null;
let compiledFor: Accelerator | null = null;
let running = false;

/** Load the LiteRT WASM runtime once; try JSPI first, fall back to plain. */
function initLitert(config: DemoConfig): Promise<boolean> {
  litertReady ??= (async (): Promise<boolean> => {
    const wasmBase = new URL(config.wasmBaseUrl, document.baseURI).toString();
    try {
      await loadLiteRt(wasmBase, { jspi: true });
      jspiActive = true;
      return true;
    } catch {
      await loadLiteRt(wasmBase, { jspi: false });
      jspiActive = false;
      return true;
    }
  })();
  return litertReady;
}

function makeTensors(inputs: SampleInput[]): Tensor[] {
  const tensors: Tensor[] = [];
  try {
    for (const input of inputs) {
      tensors.push(Tensor.fromTypedArray(input.data, input.shape));
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

function setState(state: 'loading' | 'running' | 'done' | 'error'): void {
  document.body.dataset['demoState'] = state;
}

/** Returns true when the run completed; false on failure or when busy. */
async function runDemo(config: DemoConfig, accelerator: Accelerator): Promise<boolean> {
  if (running) {
    return false;
  }
  running = true;
  const status = byId<HTMLElement>('live-status');
  const backendEl = byId<HTMLElement>('live-backend');
  const latencyEl = byId<HTMLElement>('live-latency');
  try {
    setState('loading');
    status.textContent = 'Loading LiteRT.js runtime…';
    await initLitert(config);

    if (compiledModel === null || compiledFor !== accelerator) {
      if (compiledModel !== null) {
        compiledModel.delete();
        compiledModel = null;
        compiledFor = null;
      }
      status.textContent = `Downloading and compiling model for ${accelerator} — the .tflite is fetched from its source URL, not from this site…`;
      compiledModel = await loadAndCompile(config.modelUrl, { accelerator });
      compiledFor = accelerator;
    }
    const model = compiledModel;

    setState('running');
    status.textContent = `Running ${String(config.warmupRuns)} warmup + ${String(config.timedRuns)} timed inferences…`;
    const inputs = buildSampleInputs(config.inputs, config.inputSeed);
    const inputTensors = makeTensors(inputs);
    try {
      for (let i = 0; i < config.warmupRuns; i++) {
        const outputs = await model.run(inputTensors);
        if (outputs.length > 0) {
          await outputs[0]!.data();
        }
        deleteAll(outputs);
      }
      const latencies: number[] = [];
      for (let i = 0; i < config.timedRuns; i++) {
        const start = performance.now();
        const outputs = await model.run(inputTensors);
        for (const output of outputs) {
          await output.data();
        }
        latencies.push(performance.now() - start);
        deleteAll(outputs);
      }
      const acceleratedNote =
        accelerator === 'webgpu'
          ? model.isFullyAccelerated
            ? ', fully delegated'
            : ', partial/whole fallback'
          : '';
      backendEl.textContent = `${accelerator}${jspiActive ? ' (JSPI)' : ''}${acceleratedNote}`;
      latencyEl.textContent = `${String(p50(latencies))} ms`;
      status.textContent =
        'Done. Latency is p50 of the timed runs, measured in YOUR browser on YOUR hardware — compare with the env-labeled sweep numbers below.';
      setState('done');
      measure(`run:${accelerator}`);
      return true;
    } finally {
      deleteAll(inputTensors);
    }
  } catch (err) {
    backendEl.textContent = '—';
    latencyEl.textContent = '—';
    status.textContent = `Failed on ${accelerator}: ${errorText(err)}`;
    setState('error');
    return false;
  } finally {
    running = false;
  }
}

function wireCta(config: DemoConfig): void {
  const button = byId<HTMLButtonElement>('cta-copy');
  button.addEventListener('click', () => {
    const done = (): void => {
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = 'Copy prompt';
      }, 1500);
      measure('cta-copy');
    };
    navigator.clipboard.writeText(config.ctaPrompt).then(done, () => {
      // Clipboard API unavailable (permissions/insecure context): select the
      // visible prompt text so the user can copy manually.
      const promptEl = byId<HTMLElement>('cta-prompt');
      const range = document.createRange();
      range.selectNodeContents(promptEl);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    });
  });
}

function main(): void {
  const config = readConfig();
  measure('page-load');
  const wasmButton = byId<HTMLButtonElement>('run-wasm');
  const webgpuButton = byId<HTMLButtonElement>('run-webgpu');
  wasmButton.addEventListener('click', () => {
    void runDemo(config, 'wasm');
  });
  const webgpuAvailable = isWebGPUSupported();
  if (webgpuAvailable) {
    webgpuButton.addEventListener('click', () => {
      void runDemo(config, 'webgpu');
    });
  } else {
    webgpuButton.disabled = true;
    webgpuButton.title = 'WebGPU is not available in this browser';
  }
  wireCta(config);
  // Auto-run on load: prefer WebGPU when the visitor's browser has it, and
  // fall back to WASM if the WebGPU attempt fails (flaky adapters are real).
  void (async (): Promise<void> => {
    if (webgpuAvailable) {
      const ok = await runDemo(config, 'webgpu');
      if (ok) {
        return;
      }
    }
    await runDemo(config, 'wasm');
  })();
}

main();
