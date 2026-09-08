/**
 * Types shared between the Node driver and the in-page harness.
 *
 * Everything crossing the page boundary is plain JSON-serializable data:
 * Playwright `evaluate` structured-clones arguments and return values.
 */

/** Fixture input tensors, fully materialized (deterministic, seeded in Node). */
export interface FixtureInput {
  shape: number[];
  dtype: 'float32' | 'int32';
  data: number[];
}

export type PageAccelerator = 'wasm' | 'webgpu' | 'webnn';

export interface InitResult {
  ok: boolean;
  jspi: boolean;
  error: string | null;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface ProbeResult {
  webgpuSupported: boolean;
  adapter: AdapterInfo | null;
}

export interface LoadResult {
  ok: boolean;
  /** CompiledModel.isFullyAccelerated; null when compilation failed. */
  fullyAccelerated: boolean | null;
  error: string | null;
}

export interface RunResult {
  ok: boolean;
  /** Wall time of each timed inference (run + output readback), ms. */
  latenciesMs: number[];
  /** Output tensors of the final timed run, as plain arrays. */
  outputs: number[][];
  error: string | null;
}

/** The API the bundled harness attaches to `window`. */
export interface SweepPageApi {
  init(wasmBaseUrl: string, jspi: boolean): Promise<InitResult>;
  probe(): Promise<ProbeResult>;
  loadModel(url: string, accelerator: PageAccelerator): Promise<LoadResult>;
  runModel(inputs: FixtureInput[], warmupRuns: number, timedRuns: number): Promise<RunResult>;
}

declare global {
  interface Window {
    litertSweep: SweepPageApi;
  }
}
