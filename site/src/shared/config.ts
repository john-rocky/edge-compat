/**
 * Types shared between the Node-side generator and the in-page demo runtime.
 *
 * A DemoConfig is embedded in each demo page as a JSON `<script>` block at
 * build time; everything in it must be plain JSON-serializable data.
 */

export interface TensorSpec {
  shape: number[];
  dtype: 'float32' | 'int32';
}

/**
 * One backend row of the env-labeled sweep table shown next to the live run.
 * Values come verbatim from the card's `browser.backends` records (Phase 6);
 * `status` uses the index vocabulary (DECISIONS #62):
 * load_failed | run_failed | output_mismatch | fallback | pass.
 */
export interface SweepDisplayRecord {
  backend: string;
  status: string;
  fullDelegation: boolean | null;
  outputMatch: boolean | null;
  latencyP50Ms: number | null;
  /** machine label · browser + version (+ headless) · @litertjs/core version */
  envLabel: string;
  date: string;
  provenance: string;
  /** The @litertjs/core version this record was verified against. */
  coreVersion: string;
  /**
   * The latest known release when this record lags it (Phase 11 staleness
   * surfacing, computed at build time from data/releases.json); null when
   * current or when no registry is available. Display only — never a verdict.
   */
  staleAgainst: { version: string; checkedAt: string } | null;
}

export interface DemoConfig {
  modelId: string;
  /** Absolute URL the page fetches the .tflite from at runtime (never bundled). */
  modelUrl: string;
  license: string;
  sourceUrl: string;
  /** Sample-input tensor specs; data is generated in-page from inputSeed. */
  inputs: TensorSpec[];
  inputSeed: number;
  warmupRuns: number;
  timedRuns: number;
  sweep: SweepDisplayRecord[];
  /** Relative URL (from the demo page) of the @litertjs/core WASM assets. */
  wasmBaseUrl: string;
  /** Version of @litertjs/core bundled into assets/demo.js at build time. */
  litertjsVersion: string;
  /** One-line prompt for the "Build this yourself" copy button. */
  ctaPrompt: string;
  /** One-line "reproduce this with an agent" prompt (Phase 12.2): re-runs the
   * sweep verification for this model with real commands only. */
  reproPrompt: string;
  repoUrl: string | null;
  cookbookUrl: string | null;
}
