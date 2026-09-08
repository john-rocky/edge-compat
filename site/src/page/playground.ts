/**
 * "Check your model" playground runtime (Phase 9), bundled by esbuild into
 * assets/playground.js.
 *
 * HARD PRIVACY RULE (spec 9.1): the visitor's model never leaves the browser.
 * The dropped file is read locally, parsed locally, and run locally by
 * LiteRT.js; no code path here sends model bytes — or anything derived from
 * them — over the network. The only fetches this page ever makes are the
 * static site assets and the exported matrix snapshot JSON, both at page
 * init, before any model exists. The smoke test asserts this.
 *
 * All DOM content is built with createElement/textContent — model file names
 * and runtime error strings are untrusted display input.
 */
import {
  type CompiledModel,
  Tensor,
  type TensorDetails,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
} from '@litertjs/core';

import {
  type CrossRefRow,
  FAILURE_EXPLANATIONS,
  type MatrixSnapshotDoc,
  type MatrixSnapshotRef,
  type PlaygroundConfig,
  classifyLoadError,
  compareOutputs,
  crossReference,
  lintCommand,
  playgroundCtaPrompt,
} from '../shared/playground.ts';
import { type OpInventoryEntry, TfliteParseError, readOpInventory } from '../shared/tflite.ts';
import { mulberry32, p50 } from './prng.ts';

/**
 * Measurement hook: counts of checks and CTA copies only — NEVER anything
 * model-derived (spec 9.3). Deliberately a no-op: the counter choice is the
 * owner's (DECISIONS #69, repo-stats only until an analytics stance is
 * picked). No cookies, no PII, ever.
 */
function measure(_event: string): void {
  // intentionally empty
}

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`missing element #${id}`);
  }
  return element as T;
}

function readConfig(): PlaygroundConfig {
  const raw = byId<HTMLScriptElement>('playground-config').textContent ?? '';
  return JSON.parse(raw) as PlaygroundConfig;
}

function errorText(err: unknown): string {
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err);
}

function el(tag: string, className: string | null, text: string | null): HTMLElement {
  const element = document.createElement(tag);
  if (className !== null) {
    element.className = className;
  }
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

// --- LiteRT runtime -------------------------------------------------------

let litertReady: Promise<void> | null = null;
let jspiActive = false;

function initLitert(config: PlaygroundConfig): Promise<void> {
  litertReady ??= (async (): Promise<void> => {
    const wasmBase = new URL(config.wasmBaseUrl, document.baseURI).toString();
    try {
      await loadLiteRt(wasmBase, { jspi: true });
      jspiActive = true;
    } catch {
      await loadLiteRt(wasmBase, { jspi: false });
      jspiActive = false;
    }
  })();
  return litertReady;
}

// --- Console capture (delegation evidence, verbatim) ----------------------

const MAX_EVIDENCE_MESSAGES = 100;

interface ConsoleCapture {
  messages: string[];
  stop: () => void;
}

/**
 * Capture console.warn/console.error during load and run — the runtime's
 * delegation warnings are the evidence for full_delegation and failures.
 * INFO-level lines (LiteRT routes them to console.error) are dropped, same
 * as the sweep harness. Messages still reach the real console.
 */
function captureConsole(): ConsoleCapture {
  const messages: string[] = [];
  const original = { warn: console.warn, error: console.error };
  const record = (kind: 'warning' | 'error', args: unknown[]): void => {
    const text = args.map(String).join(' ');
    if (!text.startsWith('INFO:') && messages.length < MAX_EVIDENCE_MESSAGES) {
      messages.push(`${kind}: ${text}`);
    }
  };
  console.warn = (...args: unknown[]): void => {
    record('warning', args);
    original.warn.apply(console, args);
  };
  console.error = (...args: unknown[]): void => {
    record('error', args);
    original.error.apply(console, args);
  };
  return {
    messages,
    stop: (): void => {
      console.warn = original.warn;
      console.error = original.error;
    },
  };
}

// --- Random fixture inputs from the model's own input signature -----------

class UnsupportedInputError extends Error {}

interface SampleInput {
  shape: number[];
  data: Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer> | Uint8Array<ArrayBuffer>;
}

/**
 * Deterministic random inputs (same PRNG family as the sweep; v1 runs random
 * inputs only — user-supplied tensors are an owner question, DECISIONS #96).
 */
function buildRandomInputs(details: readonly TensorDetails[], seed: number): SampleInput[] {
  return details.map((detail, index) => {
    const shape = Array.from(detail.shape, Number);
    if (shape.some((dim) => dim < 1)) {
      throw new UnsupportedInputError(
        `input '${detail.name}' has a dynamic dimension (shape ${JSON.stringify(shape)})`,
      );
    }
    const count = shape.reduce((a, b) => a * b, 1);
    const rand = mulberry32(seed + index * 1000003);
    if (detail.dtype === 'float32') {
      const data = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        data[i] = Math.fround(rand() * 2 - 1);
      }
      return { shape, data };
    }
    if (detail.dtype === 'int32') {
      const data = new Int32Array(count);
      for (let i = 0; i < count; i++) {
        data[i] = Math.floor(rand() * 16);
      }
      return { shape, data };
    }
    if (detail.dtype === 'uint8') {
      const data = new Uint8Array(count);
      for (let i = 0; i < count; i++) {
        data[i] = Math.floor(rand() * 256);
      }
      return { shape, data };
    }
    throw new UnsupportedInputError(
      `input '${detail.name}' has dtype '${String(detail.dtype)}', which LiteRT.js tensors do not support`,
    );
  });
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

// --- Live check: one record per backend, sweep-equivalent fields ----------

interface LiveRecord {
  backend: string;
  loads: boolean;
  runs: boolean;
  failureClass: string | null;
  error: string | null;
  fullDelegation: boolean | null;
  outputMatch: boolean | null;
  maxAbsDiff: number | null;
  maxRelDiff: number | null;
  comparisonNote: string | null;
  latencyP50Ms: number | null;
  evidence: string[];
  outputs: number[][] | null;
}

function emptyRecord(backend: string): LiveRecord {
  return {
    backend,
    loads: false,
    runs: false,
    failureClass: null,
    error: null,
    fullDelegation: null,
    outputMatch: null,
    maxAbsDiff: null,
    maxRelDiff: null,
    comparisonNote: null,
    latencyP50Ms: null,
    evidence: [],
    outputs: null,
  };
}

async function checkBackend(
  bytes: Uint8Array,
  backend: string,
  config: PlaygroundConfig,
  reference: LiveRecord | null,
): Promise<LiveRecord> {
  const record = emptyRecord(backend);
  const accelerator = backend === 'webgpu_mldrift' ? 'webgpu' : 'wasm';
  if (accelerator === 'webgpu' && !isWebGPUSupported()) {
    record.failureClass = 'backend_unavailable';
    record.error = 'WebGPU is not supported in this browser';
    return record;
  }
  const capture = captureConsole();
  let model: CompiledModel | null = null;
  try {
    try {
      model = await loadAndCompile(bytes, { accelerator });
    } catch (err) {
      const message = errorText(err);
      record.failureClass = classifyLoadError(message);
      record.error = message;
      return record;
    }
    record.loads = true;
    if (accelerator === 'webgpu') {
      record.fullDelegation = model.isFullyAccelerated;
    }

    let inputs: SampleInput[];
    try {
      inputs = buildRandomInputs(model.getInputDetails(), config.inputSeed);
    } catch (err) {
      record.failureClass = err instanceof UnsupportedInputError ? 'unsupported_input' : 'run_error';
      record.error = errorText(err);
      return record;
    }

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
      let finalOutputs: number[][] = [];
      for (let i = 0; i < config.timedRuns; i++) {
        const start = performance.now();
        const outputs = await model.run(inputTensors);
        const outputData: number[][] = [];
        for (const output of outputs) {
          outputData.push(Array.from(await output.data(), Number));
        }
        latencies.push(performance.now() - start);
        deleteAll(outputs);
        finalOutputs = outputData;
      }
      record.runs = true;
      record.latencyP50Ms = p50(latencies);
      record.outputs = finalOutputs;
    } catch (err) {
      record.failureClass = 'run_error';
      record.error = errorText(err);
      return record;
    } finally {
      deleteAll(inputTensors);
    }

    // Output match vs the wasm_xnnpack reference (the sweep's comparator).
    if (backend !== 'wasm_xnnpack' && record.outputs !== null) {
      if (reference === null || reference.outputs === null) {
        record.comparisonNote = 'reference backend (wasm_xnnpack) did not run';
      } else {
        const comparison = compareOutputs(
          record.outputs,
          reference.outputs,
          config.toleranceAbs,
          config.toleranceRel,
        );
        record.outputMatch = comparison.match;
        record.maxAbsDiff = comparison.maxAbsDiff;
        record.maxRelDiff = comparison.maxRelDiff;
        record.comparisonNote = comparison.note;
      }
    }
    return record;
  } finally {
    model?.delete();
    capture.stop();
    record.evidence = capture.messages;
  }
}

// --- Rendering ------------------------------------------------------------

function yesNo(value: boolean | null, yes: string, no: string): string {
  return value === null ? '—' : value ? yes : no;
}

function renderRecordRow(record: LiveRecord): HTMLTableRowElement {
  const row = document.createElement('tr');
  row.dataset['backend'] = record.backend;
  row.appendChild(el('td', null, '')).appendChild(el('code', null, record.backend));
  row.appendChild(
    el('td', record.loads ? 'status-pass' : 'status-load_failed', record.loads ? 'yes' : 'no'),
  );
  row.appendChild(
    el('td', record.runs ? 'status-pass' : 'status-load_failed', record.runs ? 'yes' : 'no'),
  );
  row.appendChild(el('td', null, yesNo(record.fullDelegation, 'yes', 'no')));

  const matchCell = el('td', null, null);
  if (record.outputMatch === null) {
    matchCell.textContent = record.comparisonNote === null ? '—' : `— (${record.comparisonNote})`;
  } else {
    matchCell.appendChild(
      el(
        'span',
        record.outputMatch ? 'status-pass' : 'status-output_mismatch',
        record.outputMatch ? 'pass' : 'fail',
      ),
    );
    matchCell.appendChild(
      el(
        'span',
        'env',
        `max abs ${record.maxAbsDiff?.toExponential(2) ?? '—'} · max rel ${record.maxRelDiff?.toExponential(2) ?? '—'}`,
      ),
    );
  }
  row.appendChild(matchCell);
  row.appendChild(
    el('td', null, record.latencyP50Ms === null ? '—' : String(record.latencyP50Ms)),
  );

  const failureCell = el('td', null, null);
  if (record.failureClass === null) {
    failureCell.textContent = '—';
  } else {
    failureCell.appendChild(el('b', null, record.failureClass));
    const explanation = FAILURE_EXPLANATIONS[record.failureClass];
    if (explanation !== undefined) {
      failureCell.appendChild(el('span', 'env', explanation));
    }
    if (record.error !== null) {
      failureCell.appendChild(el('span', 'env', record.error));
    }
  }
  row.appendChild(failureCell);
  return row;
}

function renderXref(
  rows: readonly CrossRefRow[],
  columns: readonly string[],
  refs: ReadonlyMap<string, MatrixSnapshotRef>,
): void {
  const head = byId<HTMLTableRowElement>('xref-head');
  const body = byId<HTMLTableSectionElement>('xref-body');
  head.textContent = '';
  body.textContent = '';
  head.appendChild(el('th', null, 'Op'));
  head.appendChild(el('th', null, 'Nodes'));
  for (const column of columns) {
    const th = el('th', null, column);
    const ref = refs.get(column);
    if (ref !== undefined) {
      // Staleness surfacing (Phase 11): every matrix verdict column names the
      // version + date it was verified against, with a visible flag when that
      // version lags the latest known release. The verdicts are unchanged.
      th.appendChild(
        el('span', 'env', `verified against @litertjs/core ${ref.litertVersion} (${ref.generatedAt})`),
      );
      if (ref.staleAgainst !== null) {
        th.appendChild(
          el('span', 'stale-flag', `stale — latest known ${ref.staleAgainst.version}`),
        );
      }
    }
    head.appendChild(th);
  }
  for (const row of rows) {
    const tr = document.createElement('tr');
    const opCell = el('td', null, null);
    opCell.appendChild(el('code', null, row.op));
    if (row.customCode !== null) {
      opCell.appendChild(el('span', 'env', row.customCode));
    }
    tr.appendChild(opCell);
    tr.appendChild(el('td', null, String(row.count)));
    for (const cell of row.cells) {
      const td = el('td', null, null);
      if (cell.entries.length === 0) {
        // No verdict in the exported matrix => 'unknown'. Never inferred.
        td.appendChild(el('span', 'status-unknown', 'unknown'));
        if (!cell.hasSnapshot) {
          td.appendChild(el('span', 'env', 'no snapshot exported for this backend'));
        }
      } else {
        for (const entry of cell.entries) {
          const block = el('div', 'xref-entry', null);
          block.appendChild(el('span', `status-${entry.status}`, entry.status));
          const detail: string[] = [];
          if (entry.dtypes !== undefined && entry.dtypes.length > 0) {
            detail.push(entry.dtypes.join(', '));
          }
          if (entry.conditions !== undefined) {
            detail.push(entry.conditions);
          }
          detail.push(entry.provenance);
          block.appendChild(el('span', 'env', detail.join(' · ')));
          for (const hint of entry.rewrite_hints ?? []) {
            const hintText =
              hint.expected_effect === undefined
                ? `rewrite: ${hint.rewrite}`
                : `rewrite: ${hint.rewrite} → ${hint.expected_effect}`;
            block.appendChild(el('span', 'env xref-hint', hintText));
          }
          td.appendChild(block);
        }
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

async function renderEnvBlock(config: PlaygroundConfig): Promise<void> {
  let adapterLabel = 'WebGPU unavailable';
  if (isWebGPUSupported()) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      adapterLabel =
        adapter === null
          ? 'WebGPU adapter unavailable'
          : `WebGPU adapter: ${adapter.info.vendor} ${adapter.info.architecture}`.trim();
    } catch {
      adapterLabel = 'WebGPU adapter unavailable';
    }
  }
  byId<HTMLElement>('env-block').textContent =
    `Your environment: ${navigator.userAgent} · ${adapterLabel} · ` +
    `JSPI ${jspiActive ? 'active' : 'inactive'} · @litertjs/core ${config.litertjsVersion} · ` +
    `${String(config.warmupRuns)} warmup + ${String(config.timedRuns)} timed runs, random inputs (seed ${String(config.inputSeed)})`;
}

// --- Matrix snapshot loading (page init, before any model exists) ---------

async function fetchSnapshots(config: PlaygroundConfig): Promise<Map<string, MatrixSnapshotDoc>> {
  const snapshots = new Map<string, MatrixSnapshotDoc>();
  for (const ref of config.matrix) {
    try {
      const response = await fetch(new URL(ref.href, document.baseURI));
      if (!response.ok) {
        throw new Error(`HTTP ${String(response.status)}`);
      }
      snapshots.set(ref.backend, (await response.json()) as MatrixSnapshotDoc);
    } catch (err) {
      // Missing snapshot => that backend's column honestly reads 'unknown'.
      console.warn(`matrix snapshot ${ref.href} failed to load: ${errorText(err)}`);
    }
  }
  return snapshots;
}

// --- Wiring ---------------------------------------------------------------

function wireCopyButton(buttonId: string, sourceId: string, event: string): void {
  const button = byId<HTMLButtonElement>(buttonId);
  const idleLabel = button.textContent;
  button.addEventListener('click', () => {
    const text = byId<HTMLElement>(sourceId).textContent ?? '';
    const done = (): void => {
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = idleLabel;
      }, 1500);
      measure(event);
    };
    navigator.clipboard.writeText(text).then(done, () => {
      const range = document.createRange();
      range.selectNodeContents(byId<HTMLElement>(sourceId));
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    });
  });
}

function setState(state: 'idle' | 'checking' | 'done' | 'error'): void {
  document.body.dataset['playgroundState'] = state;
}

function main(): void {
  const config = readConfig();
  measure('page-load');
  setState('idle');
  const status = byId<HTMLElement>('pg-status');
  const dropZone = byId<HTMLElement>('drop-zone');
  const input = byId<HTMLInputElement>('model-input');
  const xrefColumns = [...new Set([...config.backends, ...config.matrix.map((m) => m.backend)])].sort();
  const xrefRefs = new Map(config.matrix.map((m) => [m.backend, m]));

  const snapshotsReady = fetchSnapshots(config);
  let running = false;

  async function checkModel(file: File): Promise<void> {
    if (running) {
      return;
    }
    running = true;
    try {
      setState('checking');
      status.textContent = `Reading ${file.name} locally (nothing is uploaded)…`;
      const bytes = new Uint8Array(await file.arrayBuffer());

      // Op inventory × matrix cross-reference — independent of the live run.
      const xrefSection = byId<HTMLElement>('xref-section');
      let inventory: OpInventoryEntry[] | null = null;
      try {
        inventory = readOpInventory(bytes);
        renderXref(crossReference(inventory, xrefColumns, await snapshotsReady), xrefColumns, xrefRefs);
        xrefSection.hidden = false;
      } catch (err) {
        // Parse failure: show why, then still attempt the live run — the
        // runtime's own error is a result too.
        renderXref([], xrefColumns, xrefRefs);
        xrefSection.hidden = false;
        const message =
          err instanceof TfliteParseError ? err.message : errorText(err);
        const cell = el('td', 'status-load_failed', `operator inventory unavailable — ${message}`);
        cell.setAttribute('colspan', String(2 + xrefColumns.length));
        const row = document.createElement('tr');
        row.appendChild(cell);
        byId<HTMLTableSectionElement>('xref-body').appendChild(row);
      }

      // Live run, reference backend (wasm_xnnpack) first.
      const resultsBody = byId<HTMLTableSectionElement>('results-body');
      resultsBody.textContent = '';
      byId<HTMLElement>('results-section').hidden = false;
      status.textContent = 'Loading the LiteRT.js runtime…';
      await initLitert(config);

      const records: LiveRecord[] = [];
      let reference: LiveRecord | null = null;
      for (const backend of config.backends) {
        status.textContent = `Checking on ${backend}…`;
        const record = await checkBackend(bytes, backend, config, reference);
        if (backend === 'wasm_xnnpack') {
          reference = record;
        }
        records.push(record);
        resultsBody.appendChild(renderRecordRow(record));
      }
      await renderEnvBlock(config);

      const evidence = records.flatMap((record) =>
        record.evidence.map((line) => `[${record.backend}] ${line}`),
      );
      const evidenceDetails = byId<HTMLDetailsElement>('evidence-details');
      evidenceDetails.hidden = evidence.length === 0;
      byId<HTMLElement>('evidence-pre').textContent = evidence.join('\n');

      // CTA (spec 9.3): local lint command + cookbook prompt.
      const webgpuSnapshot = config.matrix.find((m) => m.backend === 'webgpu_mldrift') ?? null;
      byId<HTMLElement>('lint-command').textContent = lintCommand(
        file.name,
        'webgpu_mldrift',
        webgpuSnapshot,
      );
      byId<HTMLElement>('cta-prompt').textContent = playgroundCtaPrompt(file.name);
      byId<HTMLElement>('cta-section').hidden = false;

      const ran = records.filter((record) => record.runs).length;
      status.textContent =
        `Done: ${file.name} ran on ${String(ran)} of ${String(records.length)} backend(s), ` +
        'entirely in your browser. Results above are your hardware, your numbers.';
      setState('done');
      measure('model-checked');
    } catch (err) {
      status.textContent = `Check failed: ${errorText(err)}`;
      setState('error');
    } finally {
      running = false;
    }
  }

  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file !== undefined) {
      void checkModel(file);
    }
  });
  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-active');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
  });
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-active');
    const file = event.dataTransfer?.files[0];
    if (file !== undefined) {
      void checkModel(file);
    }
  });
  wireCopyButton('lint-copy', 'lint-command', 'lint-copy');
  wireCopyButton('cta-copy', 'cta-prompt', 'cta-copy');
  status.textContent = 'Ready — drop a .tflite file above.';
}

main();
