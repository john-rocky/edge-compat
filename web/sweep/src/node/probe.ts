/**
 * Single-fixture probe driver for the Phase 8 `webgpu_playwright` runner.
 *
 * `node --experimental-strip-types src/node/probe.ts --job <job.json>` reads
 * one job — `{modelPath, accelerator, inputs, timeoutMs}` — runs the fixture
 * once through the sweep's in-page harness on headless Chromium, and prints
 * exactly one JSON report line to stdout.
 *
 * Numeric honesty: the input tensors are fully materialized by the CALLER
 * (the Python runner mirrors its CPU-reference feeds into the job file), so
 * the browser computes on byte-identical inputs — this driver never invents
 * inputs. Outputs cross back verbatim as plain arrays; the numeric comparison
 * happens caller-side against the same CPU reference.
 *
 * Exit codes: 0 = a report line was printed (load/run failures ARE
 * measurements and report ok:false with their stage); 1 = internal error
 * before a report could be produced (nothing measured, nothing printed).
 */
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

import { type Browser, chromium } from 'playwright';

import type { AdapterInfo, FixtureInput, RunResult } from '../shared/protocol.ts';
import { type SweepServer, startServer } from './server.ts';

const require = createRequire(import.meta.url);

const MAX_EVIDENCE_MESSAGES = 8;

export type ProbeAccelerator = 'wasm' | 'webgpu';

export interface ProbeJob {
  modelPath: string;
  accelerator: ProbeAccelerator;
  inputs: FixtureInput[];
  timeoutMs: number;
}

/** One-line report printed to stdout; `stage` names the first failing step. */
export interface ProbeReport {
  ok: boolean;
  stage: 'init' | 'probe' | 'load' | 'run' | null;
  error: string | null;
  coreVersion: string;
  chromiumVersion: string | null;
  jspi: boolean | null;
  webgpuSupported: boolean | null;
  adapter: AdapterInfo | null;
  /** CompiledModel.isFullyAccelerated; null on wasm or when load failed. */
  fullyAccelerated: boolean | null;
  /** Output tensors as flat arrays; null unless the run completed. */
  outputs: number[][] | null;
  /** Console warnings/errors and page errors, verbatim (INFO lines dropped). */
  evidence: string[];
}

export function litertCoreVersion(): string {
  const pkg = require('@litertjs/core/package.json') as { version: string };
  return pkg.version;
}

function isFixtureInput(value: unknown): value is FixtureInput {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const input = value as Record<string, unknown>;
  return (
    Array.isArray(input['shape']) &&
    input['shape'].every((dim) => typeof dim === 'number' && Number.isInteger(dim) && dim >= 1) &&
    (input['dtype'] === 'float32' || input['dtype'] === 'int32') &&
    Array.isArray(input['data']) &&
    input['data'].every((v) => typeof v === 'number' && Number.isFinite(v))
  );
}

/** Validate a parsed job file. Throws with a precise reason on any mismatch. */
export function parseJob(raw: unknown): ProbeJob {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('job must be a JSON object');
  }
  const job = raw as Record<string, unknown>;
  if (typeof job['modelPath'] !== 'string' || job['modelPath'] === '') {
    throw new Error('job.modelPath must be a non-empty string');
  }
  if (job['accelerator'] !== 'wasm' && job['accelerator'] !== 'webgpu') {
    throw new Error(`job.accelerator must be 'wasm' | 'webgpu', got ${JSON.stringify(job['accelerator'])}`);
  }
  if (!Array.isArray(job['inputs']) || job['inputs'].length === 0) {
    throw new Error('job.inputs must be a non-empty array');
  }
  if (!job['inputs'].every(isFixtureInput)) {
    throw new Error('job.inputs entries must be {shape: int[]>=1, dtype: float32|int32, data: finite number[]}');
  }
  const timeout = job['timeoutMs'];
  if (typeof timeout !== 'number' || !Number.isInteger(timeout) || timeout < 1000) {
    throw new Error('job.timeoutMs must be an integer >= 1000');
  }
  return {
    modelPath: job['modelPath'],
    accelerator: job['accelerator'],
    inputs: job['inputs'],
    timeoutMs: timeout,
  };
}

function withTimeout<T>(promise: Promise<T>, ms: number, stage: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${stage} timed out after ${String(ms)}ms`));
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

async function measure(job: ProbeJob, browser: Browser, baseUrl: string): Promise<ProbeReport> {
  const report: ProbeReport = {
    ok: false,
    stage: null,
    error: null,
    coreVersion: litertCoreVersion(),
    chromiumVersion: browser.version(),
    jspi: null,
    webgpuSupported: null,
    adapter: null,
    fullyAccelerated: null,
    outputs: null,
    evidence: [],
  };
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    page.on('console', (message) => {
      const kind = message.type();
      const text = message.text();
      // Mirror the sweep: the runtime logs INFO lines (nondeterministic
      // pointers) to console.error; warnings and real errors are evidence.
      if (text.startsWith('INFO:')) {
        return;
      }
      if ((kind === 'warning' || kind === 'error') && report.evidence.length < MAX_EVIDENCE_MESSAGES) {
        report.evidence.push(`${kind}: ${text}`);
      }
    });
    page.on('pageerror', (err) => {
      if (report.evidence.length < MAX_EVIDENCE_MESSAGES) {
        report.evidence.push(`pageerror: ${err.message}`);
      }
    });
    await page.goto(`${baseUrl}/`, { waitUntil: 'load' });

    const init = await withTimeout(
      page.evaluate(
        (args: { wasmBaseUrl: string; jspi: boolean }) =>
          window.litertSweep.init(args.wasmBaseUrl, args.jspi),
        { wasmBaseUrl: `${baseUrl}/wasm/`, jspi: true },
      ),
      job.timeoutMs,
      'init',
    );
    report.jspi = init.jspi;
    if (!init.ok) {
      report.stage = 'init';
      report.error = init.error;
      return report;
    }

    const probed = await withTimeout(
      page.evaluate(() => window.litertSweep.probe()),
      job.timeoutMs,
      'probe',
    );
    report.webgpuSupported = probed.webgpuSupported;
    report.adapter = probed.adapter;
    if (job.accelerator === 'webgpu' && !probed.webgpuSupported) {
      report.stage = 'probe';
      report.error = 'WebGPU is not supported in this browser environment';
      return report;
    }

    const load = await withTimeout(
      page.evaluate(
        (args: { url: string; accelerator: 'wasm' | 'webgpu' }) =>
          window.litertSweep.loadModel(args.url, args.accelerator),
        { url: `${baseUrl}/models/probe`, accelerator: job.accelerator },
      ),
      job.timeoutMs,
      'load',
    );
    if (!load.ok) {
      report.stage = 'load';
      report.error = load.error;
      return report;
    }
    if (job.accelerator === 'webgpu') {
      report.fullyAccelerated = load.fullyAccelerated;
    }

    const run: RunResult = await withTimeout(
      page.evaluate(
        (args: { inputs: FixtureInput[]; warmup: number; timed: number }) =>
          window.litertSweep.runModel(args.inputs, args.warmup, args.timed),
        { inputs: job.inputs, warmup: 1, timed: 1 },
      ),
      job.timeoutMs,
      'run',
    );
    if (!run.ok) {
      report.stage = 'run';
      report.error = run.error;
      return report;
    }
    report.outputs = run.outputs;
    report.ok = true;
    return report;
  } finally {
    await context.close();
  }
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: { job: { type: 'string' } },
    strict: true,
  });
  if (values.job === undefined) {
    process.stderr.write('error: --job <file> is required\n');
    process.exit(1);
  }
  let job: ProbeJob;
  try {
    job = parseJob(JSON.parse(fs.readFileSync(values.job, 'utf-8')));
    if (!fs.existsSync(job.modelPath)) {
      throw new Error(`model file ${job.modelPath} does not exist`);
    }
  } catch (err) {
    process.stderr.write(`error: invalid job: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  }

  const server: SweepServer = await startServer(new Map([['probe', job.modelPath]]));
  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({
      headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-features=WebAssemblyJSPI', '--use-angle=metal'],
    });
    const report = await measure(job, browser, server.baseUrl);
    process.stdout.write(`${JSON.stringify(report)}\n`);
  } finally {
    if (browser !== null) {
      await browser.close();
    }
    await server.close();
  }
}

const invokedDirectly =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) {
  main().catch((err: unknown) => {
    process.stderr.write(`error: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}
