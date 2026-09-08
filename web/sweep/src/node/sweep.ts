/**
 * Batch sweep orchestrator: for each catalog entry x backend, drive the
 * in-page harness in a fresh Playwright browser context. One model crashing
 * or hanging its page must never kill the batch — the crash IS the result.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

import { type Browser, type BrowserContext, type Page, chromium } from 'playwright';

import type { FixtureInput, RunResult } from '../shared/protocol.ts';
import { cachePathFor, ensureCached, trimCache } from './cache.ts';
import {
  type CatalogEntry,
  CatalogError,
  isFixtureFileSpec,
  parseCatalog,
  parseShapeSpec,
} from './catalog.ts';
import { type SweepEnv, buildEnv } from './env.ts';
import { buildFixtureInputs, loadFixtureFile } from './inputs.ts';
import { canonicalStringify, compareOutputs, p50 } from './results.ts';
import { type SweepServer, startServer } from './server.ts';
import { validateSweepResult } from './validate.ts';

export type BackendId = 'wasm_xnnpack' | 'webgpu_mldrift' | 'webnn';

const BACKEND_ACCELERATOR = {
  wasm_xnnpack: 'wasm',
  webgpu_mldrift: 'webgpu',
  webnn: 'webnn',
} as const;

/** The wasm_xnnpack reference runs first; output order is alphabetical anyway. */
const BACKEND_ORDER: readonly BackendId[] = ['wasm_xnnpack', 'webgpu_mldrift', 'webnn'];

export interface SweepConfig {
  warmup_runs: number;
  timed_runs: number;
  tolerance_abs: number;
  tolerance_rel: number;
  input_seed: number;
}

export interface SweepOptions {
  catalogPath: string;
  outDir: string;
  experimental: boolean;
  headed: boolean;
  machineLabel: string;
  date: string;
  provenance: 'measured' | 'example';
  timeoutMs: number;
  config: SweepConfig;
  /**
   * When set, URL-sourced models are downloaded into this directory — one
   * model at a time, right before its sweep — and served from the local
   * server; a later run of an unchanged catalog hits the cache (Phase 11
   * re-sweep caching). The recorded `source` stays the catalog URL either way.
   */
  cacheDir?: string | undefined;
  /**
   * With `cacheDir`: around each model, evict least-recently-used cached
   * models until the directory holds at most this many bytes. 0 keeps
   * nothing between models — the CI setting, where the 33 GB catalog fits
   * neither the runner's disk nor GitHub's 10 GB cache. Unset = unbounded.
   */
  cacheMaxBytes?: number | undefined;
  /** Sweep only these catalog model_ids; every id must be in the catalog. */
  only?: readonly string[] | undefined;
  /** Skip models whose `<out>/<model_id>.json` already exists (resume a run). */
  skipExisting?: boolean | undefined;
}

export interface BackendRecord {
  backend: BackendId;
  loads: boolean;
  runs: boolean;
  failure_class: string | null;
  error: string | null;
  full_delegation: boolean | null;
  output_match: boolean | null;
  max_abs_diff: number | null;
  max_rel_diff: number | null;
  latency_p50_ms: number | null;
  delegation_evidence: string[];
  env: SweepEnv;
  date: string;
  provenance: 'measured' | 'example';
}

export interface SweepResultFile {
  schema_version: '1.0';
  model_id: string;
  source: string;
  license: string;
  input_spec: string;
  notes: string;
  config: SweepConfig;
  results: BackendRecord[];
}

class TimeoutError extends Error {}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(`${label} timed out after ${String(ms)}ms`));
    }, ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err: unknown) => {
        clearTimeout(timer);
        reject(err instanceof Error ? err : new Error(String(err)));
      },
    );
  });
}

function classifyLoadError(message: string): string {
  const lower = message.toLowerCase();
  if (/(404|failed to fetch|networkerror|net::err)/.test(lower)) {
    return 'fetch_failed';
  }
  if (/(out of memory|memory access out of bounds|cannot allocate|allocation failed|oom)/.test(lower)) {
    return 'wasm_memory_ceiling';
  }
  if (/(flatbuffer|verif|invalid model|parse|malformed|not a valid|failed to load model)/.test(lower)) {
    return 'parse_error';
  }
  return 'compile_error';
}

const MAX_EVIDENCE_MESSAGES = 100;

interface PageHandle {
  context: BrowserContext;
  page: Page;
  evidence: string[];
  crashed: { value: boolean };
}

async function openPage(browser: Browser, baseUrl: string): Promise<PageHandle> {
  const context = await browser.newContext();
  const page = await context.newPage();
  const evidence: string[] = [];
  const crashed = { value: false };
  page.on('console', (message) => {
    const kind = message.type();
    const text = message.text();
    // The LiteRT runtime routes its INFO-level logging to console.error;
    // those lines carry pointer addresses (nondeterministic) and are not
    // delegation evidence. Warnings and real errors are kept verbatim.
    if (text.startsWith('INFO:')) {
      return;
    }
    if ((kind === 'warning' || kind === 'error') && evidence.length < MAX_EVIDENCE_MESSAGES) {
      evidence.push(`${kind}: ${text}`);
    }
  });
  page.on('pageerror', (err) => {
    if (evidence.length < MAX_EVIDENCE_MESSAGES) {
      evidence.push(`pageerror: ${err.message}`);
    }
  });
  page.on('crash', () => {
    crashed.value = true;
  });
  await page.goto(`${baseUrl}/`, { waitUntil: 'load' });
  return { context, page, evidence, crashed };
}

interface BackendMeasurement {
  record: BackendRecord;
  outputs: number[][] | null;
}

async function measureBackend(params: {
  browser: Browser;
  baseUrl: string;
  browserVersion: string;
  entry: CatalogEntry;
  backend: BackendId;
  inputs: FixtureInput[];
  referenceOutputs: number[][] | null;
  opts: SweepOptions;
  /** True when the model is served by the local server (path source or cached download). */
  servedLocally: boolean;
}): Promise<BackendMeasurement> {
  const { browser, baseUrl, entry, backend, inputs, referenceOutputs, opts } = params;
  const accelerator = BACKEND_ACCELERATOR[backend];
  const modelUrl = params.servedLocally ? `${baseUrl}/models/${entry.modelId}` : entry.source;

  const record: BackendRecord = {
    backend,
    loads: false,
    runs: false,
    failure_class: null,
    error: null,
    full_delegation: null,
    output_match: null,
    max_abs_diff: null,
    max_rel_diff: null,
    latency_p50_ms: null,
    delegation_evidence: [],
    env: buildEnv({
      browserVersion: params.browserVersion,
      headless: !opts.headed,
      jspi: false,
      adapter: null,
      machineLabel: opts.machineLabel,
    }),
    date: opts.date,
    provenance: opts.provenance,
  };
  let outputs: number[][] | null = null;

  let handle: PageHandle | null = null;
  try {
    handle = await openPage(browser, baseUrl);
    const { page } = handle;

    const init = await withTimeout(
      page.evaluate((jspi) => window.litertSweep.init('/wasm/', jspi), true),
      opts.timeoutMs,
      'init',
    );
    record.env.jspi = init.jspi;
    if (!init.ok) {
      record.failure_class = 'init_failed';
      record.error = init.error;
      return { record, outputs };
    }

    const probeResult = await withTimeout(
      page.evaluate(() => window.litertSweep.probe()),
      opts.timeoutMs,
      'probe',
    );
    record.env.webgpu_adapter = probeResult.adapter;
    if (accelerator === 'webgpu' && !probeResult.webgpuSupported) {
      record.failure_class = 'backend_unavailable';
      record.error = 'WebGPU is not supported in this browser environment';
      return { record, outputs };
    }

    const load = await withTimeout(
      page.evaluate(
        (args: { url: string; accelerator: 'wasm' | 'webgpu' | 'webnn' }) =>
          window.litertSweep.loadModel(args.url, args.accelerator),
        { url: modelUrl, accelerator },
      ),
      opts.timeoutMs,
      'load',
    );
    if (!load.ok) {
      record.failure_class = classifyLoadError(load.error ?? '');
      record.error = load.error;
      return { record, outputs };
    }
    record.loads = true;
    if (accelerator !== 'wasm') {
      record.full_delegation = load.fullyAccelerated;
    }

    const run: RunResult = await withTimeout(
      page.evaluate(
        (args: { inputs: FixtureInput[]; warmup: number; timed: number }) =>
          window.litertSweep.runModel(args.inputs, args.warmup, args.timed),
        { inputs, warmup: opts.config.warmup_runs, timed: opts.config.timed_runs },
      ),
      opts.timeoutMs,
      'run',
    );
    if (!run.ok) {
      record.failure_class = 'run_error';
      record.error = run.error;
      return { record, outputs };
    }
    // A run can resolve with outputs while the backend logged a fatal
    // preparation error ("Node ... failed to prepare.", "failed to create
    // XNNPACK runtime") — the buffers were never computed and the sub-ms
    // latency is an error path, not a measurement.
    const fatal = (handle?.evidence ?? []).find((line) =>
      /failed to prepare|failed to create .* runtime/i.test(line),
    );
    if (fatal !== undefined) {
      record.failure_class = 'backend_error';
      record.error = `backend reported a fatal error during run: ${fatal}`;
      return { record, outputs };
    }
    record.runs = true;
    record.latency_p50_ms = p50(run.latenciesMs);
    outputs = run.outputs;

    if (backend !== 'wasm_xnnpack' && referenceOutputs !== null) {
      const comparison = compareOutputs(
        run.outputs,
        referenceOutputs,
        opts.config.tolerance_abs,
        opts.config.tolerance_rel,
      );
      record.output_match = comparison.match;
      record.max_abs_diff = comparison.maxAbsDiff;
      record.max_rel_diff = comparison.maxRelDiff;
      if (comparison.note !== null) {
        record.error = comparison.note;
      }
    }
    return { record, outputs };
  } catch (err) {
    if (handle?.crashed.value === true) {
      record.failure_class = 'page_crash';
      record.error = 'browser page crashed';
    } else if (err instanceof TimeoutError) {
      record.failure_class = 'timeout';
      record.error = err.message;
    } else {
      record.failure_class = record.loads ? 'run_error' : 'compile_error';
      record.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    }
    return { record, outputs };
  } finally {
    if (handle !== null) {
      record.delegation_evidence = handle.evidence;
      await handle.context.close().catch(() => undefined);
    }
  }
}

function resolveInputs(entry: CatalogEntry, catalogDir: string, seed: number): FixtureInput[] {
  if (isFixtureFileSpec(entry.inputSpec)) {
    return loadFixtureFile(path.resolve(catalogDir, entry.inputSpec));
  }
  return buildFixtureInputs(parseShapeSpec(entry.inputSpec), seed);
}

export interface SweepRunSummary {
  files: string[];
  validationFailures: string[];
  exitCode: 0 | 1;
}

export async function runSweep(opts: SweepOptions): Promise<SweepRunSummary> {
  const catalogDir = path.dirname(path.resolve(opts.catalogPath));
  const catalogText = fs.readFileSync(opts.catalogPath, 'utf8');
  let entries = parseCatalog(catalogText, catalogDir);
  if (opts.only !== undefined) {
    const known = new Set(entries.map((entry) => entry.modelId));
    const unknown = opts.only.filter((id) => !known.has(id));
    if (unknown.length > 0) {
      throw new CatalogError(`--only names model(s) not in the catalog: ${unknown.join(', ')}`);
    }
    const wanted = new Set(opts.only);
    entries = entries.filter((entry) => wanted.has(entry.modelId));
  }
  if (opts.skipExisting === true) {
    entries = entries.filter((entry) => {
      const exists = fs.existsSync(path.join(opts.outDir, `${entry.modelId}.json`));
      if (exists) {
        process.stdout.write(`skip ${entry.modelId} (result exists)\n`);
      }
      return !exists;
    });
  }

  // Fixture inputs resolve up front: a broken input_spec is a manifest defect
  // (usage error), not a sweep result.
  const inputsByModel = new Map<string, FixtureInput[]>();
  for (const entry of entries) {
    inputsByModel.set(entry.modelId, resolveInputs(entry, catalogDir, opts.config.input_seed));
  }

  const backends: BackendId[] = opts.experimental
    ? [...BACKEND_ORDER]
    : BACKEND_ORDER.filter((backend) => backend !== 'webnn');

  const modelRoutes = new Map<string, string>();
  for (const entry of entries) {
    if (!entry.sourceIsUrl) {
      modelRoutes.set(entry.modelId, entry.source);
    }
  }
  // URL-sourced models are staged one at a time, right before their sweep,
  // so the disk never holds more than the cache budget plus the model in
  // use (pre-downloading the whole 33 GB catalog filled a CI runner's disk
  // on 2026-08-27). A failed download falls back to the direct URL so the
  // page can record the honest fetch_failed result instead of the batch
  // aborting.
  const stage = async (entry: CatalogEntry): Promise<void> => {
    if (!entry.sourceIsUrl || opts.cacheDir === undefined) {
      return;
    }
    try {
      const cached = await ensureCached(entry.source, opts.cacheDir);
      modelRoutes.set(entry.modelId, cached.path);
      process.stdout.write(
        `cache ${cached.hit ? 'hit ' : 'miss'} ${entry.modelId} (${entry.source})\n`,
      );
    } catch (err) {
      process.stderr.write(
        `cache: download failed for ${entry.modelId}, falling back to direct fetch: ` +
          `${err instanceof Error ? err.message : String(err)}\n`,
      );
    }
  };
  const trim = (keep: string | undefined): void => {
    if (opts.cacheDir === undefined || opts.cacheMaxBytes === undefined) {
      return;
    }
    for (const evicted of trimCache(opts.cacheDir, opts.cacheMaxBytes, keep)) {
      process.stdout.write(`cache evict ${path.basename(evicted)}\n`);
    }
  };

  fs.mkdirSync(opts.outDir, { recursive: true });

  const files: string[] = [];
  const validationFailures: string[] = [];
  const summaryRows: string[][] = [];
  // Everything that keeps the event loop alive is closed in `finally`,
  // whichever step throws: a browser that fails to launch must not leave the
  // server listening and the process hanging until a CI job timeout (the
  // 2026-08-27 run spent five hours that way).
  const server: SweepServer = await startServer(modelRoutes);
  let browser: Browser | undefined;
  try {
    browser = await chromium.launch({
      headless: !opts.headed,
      args: ['--enable-unsafe-webgpu', '--enable-features=WebAssemblyJSPI', '--use-angle=metal'],
    });
    const browserVersion = browser.version();

    for (const entry of entries) {
      trim(
        entry.sourceIsUrl && opts.cacheDir !== undefined
          ? cachePathFor(opts.cacheDir, entry.source)
          : undefined,
      );
      await stage(entry);
      const inputs = inputsByModel.get(entry.modelId) ?? [];
      let referenceOutputs: number[][] | null = null;
      const records: BackendRecord[] = [];
      for (const backend of backends) {
        const { record, outputs } = await measureBackend({
          browser,
          baseUrl: server.baseUrl,
          browserVersion,
          entry,
          backend,
          inputs,
          referenceOutputs,
          opts,
          servedLocally: modelRoutes.has(entry.modelId),
        });
        if (backend === 'wasm_xnnpack' && outputs !== null) {
          referenceOutputs = outputs;
        }
        records.push(record);
        summaryRows.push([
          entry.modelId,
          backend,
          record.loads ? 'yes' : 'no',
          record.runs ? 'yes' : 'no',
          record.full_delegation === null ? '-' : record.full_delegation ? 'full' : 'partial',
          record.output_match === null ? '-' : record.output_match ? 'pass' : 'FAIL',
          record.latency_p50_ms === null ? '-' : record.latency_p50_ms.toFixed(3),
          record.failure_class ?? '-',
        ]);
      }
      records.sort((a, b) => a.backend.localeCompare(b.backend));

      const resultFile: SweepResultFile = {
        schema_version: '1.0',
        model_id: entry.modelId,
        source: entry.rawSource,
        license: entry.license,
        input_spec: entry.inputSpec,
        notes: entry.notes,
        config: opts.config,
        results: records,
      };
      const outPath = path.join(opts.outDir, `${entry.modelId}.json`);
      const errors = validateSweepResult(resultFile);
      if (errors.length > 0) {
        validationFailures.push(`${entry.modelId}: ${errors.join('; ')}`);
      }
      fs.writeFileSync(outPath, canonicalStringify(resultFile));
      files.push(outPath);
      trim(undefined);
    }
  } finally {
    await browser?.close().catch(() => undefined);
    await server.close().catch(() => undefined);
  }

  printSummary(summaryRows);
  if (validationFailures.length > 0) {
    process.stderr.write(`\nschema validation FAILED for ${String(validationFailures.length)} result file(s):\n`);
    for (const failure of validationFailures) {
      process.stderr.write(`  ${failure}\n`);
    }
  }
  return { files, validationFailures, exitCode: validationFailures.length > 0 ? 1 : 0 };
}

function printSummary(rows: string[][]): void {
  const header = ['model', 'backend', 'loads', 'runs', 'delegation', 'match', 'p50_ms', 'failure'];
  const table = [header, ...rows];
  const widths = header.map((_, col) => Math.max(...table.map((row) => (row[col] ?? '').length)));
  for (const row of table) {
    const line = row.map((cell, col) => (cell ?? '').padEnd(widths[col] ?? 0)).join('  ');
    process.stdout.write(`${line.trimEnd()}\n`);
  }
}
