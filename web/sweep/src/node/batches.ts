/**
 * Batch driver for a full-catalog sweep: runs the harness in a FRESH process
 * per `--batch` models, because the runner accumulates memory across the
 * models of one run (README "Traps": a 36-model batch died at 27 with the
 * heap exhausted; the same models one at a time all landed). A batch whose
 * process dies is re-run one model at a time; a model that kills a process
 * on its own is reported as UNMEASURED and the driver exits 1 — the output
 * directory still holds every model that was measured, and a re-run skips
 * them. This is what the weekly CI resweep runs.
 *
 * With `--history <dir>` (earlier dated snapshots of the same runtime
 * version) the pending models run stalest first: never measured there, then
 * by the date of their newest snapshot, oldest first. With
 * `--budget-minutes <m>` no harness run starts after `m` minutes; the models
 * not reached are printed as DEFERRED (exit status unchanged), and the next
 * run takes them before this run's models. Together they let a catalog larger
 * than one CI step be covered across runs instead of the same head being
 * re-measured every week.
 * `--dry-run` prints that plan without running the harness.
 *
 * usage: node --experimental-strip-types src/node/batches.ts \
 *          --catalog <csv> --out <dir> [--batch <n>] [--history <dir>] \
 *          [--budget-minutes <m>] [--dry-run [--minutes-per-model <x>]] \
 *          [-- <harness flags...>]
 */
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { CatalogError, parseCatalog } from './catalog.ts';

export interface BatchOptions {
  catalogPath: string;
  outDir: string;
  batchSize: number;
  /** Earlier dated snapshots (`<historyDir>/<YYYY-MM-DD>/<id>.json`): sweep stalest first. */
  historyDir?: string | undefined;
  /** No harness run starts after this many minutes; the rest are deferred. */
  budgetMinutes?: number | undefined;
}

export interface BatchPlan {
  /** Catalog models, catalog order. */
  all: string[];
  /** Models without a result in `outDir`, in sweep order. */
  pending: string[];
  /** Newest snapshot date under `historyDir` for each pending model measured there. */
  lastMeasured: Map<string, string>;
}

export interface BatchOutcome {
  /** Models with a result file in `outDir` when the driver finished, catalog order. */
  measured: string[];
  /** Models still without a result after their batch run and a run alone. */
  unmeasured: string[];
  /** Models not started because the time budget ran out, sweep order. */
  deferred: string[];
  /** Harness runs that exited 1: a result file failed schema validation. */
  invalidRuns: number;
}

/** Runs the harness on exactly these models; returns its exit status (null = killed by a signal). */
export type HarnessRunner = (modelIds: readonly string[]) => number | null;

const DATE_DIR = /^\d{4}-\d{2}-\d{2}$/;

/** The newest `<historyDir>/<date>/` holding `<id>.json`, for each id that has one. */
export function lastMeasured(historyDir: string, ids: readonly string[]): Map<string, string> {
  const last = new Map<string, string>();
  if (!fs.existsSync(historyDir)) {
    return last;
  }
  const newestFirst = fs
    .readdirSync(historyDir)
    .filter((name) => DATE_DIR.test(name))
    .sort()
    .reverse();
  for (const id of ids) {
    const date = newestFirst.find((d) => fs.existsSync(path.join(historyDir, d, `${id}.json`)));
    if (date !== undefined) {
      last.set(id, date);
    }
  }
  return last;
}

export function planBatches(opts: BatchOptions): BatchPlan {
  const catalogDir = path.dirname(path.resolve(opts.catalogPath));
  const entries = parseCatalog(fs.readFileSync(opts.catalogPath, 'utf8'), catalogDir);
  const all = entries.map((entry) => entry.modelId);
  const pending = all.filter((id) => !fs.existsSync(path.join(opts.outDir, `${id}.json`)));
  if (opts.historyDir === undefined) {
    return { all, pending, lastMeasured: new Map() };
  }
  const last = lastMeasured(opts.historyDir, pending);
  // Stable sort: '' (never measured) before any date, dates oldest first,
  // catalog order among equals.
  const key = (id: string): string => last.get(id) ?? '';
  const ordered = [...pending].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
  return { all, pending: ordered, lastMeasured: last };
}

export function runBatches(
  opts: BatchOptions,
  run: HarnessRunner,
  log: (line: string) => void = (line) => process.stdout.write(`${line}\n`),
  now: () => number = Date.now,
): BatchOutcome {
  const started = now();
  const { all, pending } = planBatches(opts);
  const done = (id: string): boolean => fs.existsSync(path.join(opts.outDir, `${id}.json`));
  if (pending.length < all.length) {
    log(
      `batches: ${String(all.length - pending.length)} model(s) already have a result in ` +
        `${opts.outDir} — skipped`,
    );
  }
  const budgetMs = opts.budgetMinutes === undefined ? Infinity : opts.budgetMinutes * 60_000;
  const outOfTime = (): boolean => now() - started >= budgetMs;
  const unmeasured: string[] = [];
  const deferred: string[] = [];
  let invalidRuns = 0;
  const note = (status: number | null): void => {
    if (status === 1) {
      invalidRuns += 1;
    }
  };
  const describe = (status: number | null): string =>
    status === null ? 'by signal' : `with status ${String(status)}`;

  for (let start = 0; start < pending.length; start += opts.batchSize) {
    if (outOfTime()) {
      deferred.push(...pending.slice(start));
      break;
    }
    const batch = pending.slice(start, start + opts.batchSize);
    log(
      `batches: ${String(start + 1)}-${String(start + batch.length)} of ` +
        `${String(pending.length)}: ${batch.join(' ')}`,
    );
    const status = run(batch);
    note(status);
    const missing = batch.filter((id) => !done(id));
    if (missing.length === 0) {
      continue;
    }
    log(
      `batches: harness exited ${describe(status)} leaving ${String(missing.length)} ` +
        'model(s) without a result — retrying each alone',
    );
    for (const id of missing) {
      if (outOfTime()) {
        deferred.push(id);
        continue;
      }
      const single = run([id]);
      note(single);
      if (!done(id)) {
        unmeasured.push(id);
        log(`batches: UNMEASURED ${id} — the harness exited ${describe(single)} on it alone`);
      }
    }
  }
  if (deferred.length > 0) {
    log(
      `batches: time budget of ${String(opts.budgetMinutes)} min spent — ` +
        `${String(deferred.length)} model(s) deferred to the next run`,
    );
  }
  return { measured: all.filter(done), unmeasured, deferred, invalidRuns };
}

const USAGE =
  'usage: node --experimental-strip-types src/node/batches.ts --catalog <csv> --out <dir> ' +
  '[--batch <n>] [--history <dir>] [--budget-minutes <m>] ' +
  '[--dry-run [--minutes-per-model <x>]] [-- <harness flags...>]';

function usage(message: string): never {
  process.stderr.write(`error: ${message}\n\n${USAGE}\n`);
  process.exit(2);
}

function positiveInt(flag: string, value: string): number {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n < 1 || String(n) !== value.trim()) {
    usage(`${flag} must be an integer >= 1, got '${value}'`);
  }
  return n;
}

/** The plan only: counts, order and, given a per-model time, the expected minutes. */
function printPlan(opts: BatchOptions, minutesPerModel: number | undefined): void {
  const plan = planBatches(opts);
  const out = (line: string): void => {
    process.stdout.write(`batches (dry run): ${line}\n`);
  };
  out(
    `${String(plan.all.length)} catalog model(s), ` +
      `${String(plan.all.length - plan.pending.length)} already in ${opts.outDir}, ` +
      `${String(plan.pending.length)} pending`,
  );
  if (opts.historyDir !== undefined) {
    const dates = [...plan.lastMeasured.values()].sort();
    const never = plan.pending.length - dates.length;
    out(
      `order = stalest first under ${opts.historyDir}: ${String(never)} never measured there` +
        (dates.length > 0
          ? `, then ${String(dates.length)} by snapshot date (${dates[0] ?? ''} … ${dates[dates.length - 1] ?? ''})`
          : ''),
    );
  }
  const batches = Math.ceil(plan.pending.length / opts.batchSize);
  out(`${String(batches)} batch(es) of up to ${String(opts.batchSize)}`);
  if (minutesPerModel !== undefined) {
    const all = plan.pending.length * minutesPerModel;
    out(`at ${String(minutesPerModel)} min/model: all pending = ${all.toFixed(0)} min`);
    if (opts.budgetMinutes !== undefined) {
      // A batch that starts before the budget runs to its end.
      const batchMinutes = minutesPerModel * opts.batchSize;
      const reach = Math.min(
        plan.pending.length,
        Math.ceil(opts.budgetMinutes / batchMinutes) * opts.batchSize,
      );
      out(
        `budget ${String(opts.budgetMinutes)} min: about ${String(reach)} model(s) in ` +
          `${(reach * minutesPerModel).toFixed(0)} min this run, ` +
          `${String(plan.pending.length - reach)} deferred to the next`,
      );
    }
  } else if (opts.budgetMinutes !== undefined) {
    out(`budget ${String(opts.budgetMinutes)} min: no harness run starts after it`);
  }
  out(`first batch: ${plan.pending.slice(0, opts.batchSize).join(' ')}`);
}

function main(): void {
  let parsed;
  try {
    parsed = parseArgs({
      options: {
        catalog: { type: 'string' },
        out: { type: 'string' },
        batch: { type: 'string', default: '8' },
        history: { type: 'string' },
        'budget-minutes': { type: 'string' },
        'dry-run': { type: 'boolean', default: false },
        'minutes-per-model': { type: 'string' },
      },
      allowPositionals: true,
      strict: true,
    });
  } catch (err) {
    usage(err instanceof Error ? err.message : String(err));
  }
  const { values, positionals } = parsed;
  if (values.catalog === undefined) {
    usage('--catalog is required');
  }
  if (values.out === undefined) {
    usage('--out is required');
  }
  const catalogPath = values.catalog;
  const outDir = values.out;
  const batchSize = positiveInt('--batch', values.batch);
  const budgetFlag = values['budget-minutes'];
  const budgetMinutes =
    budgetFlag === undefined ? undefined : positiveInt('--budget-minutes', budgetFlag);
  const opts: BatchOptions = { catalogPath, outDir, batchSize, historyDir: values.history, budgetMinutes };

  if (values['dry-run']) {
    const perModelFlag = values['minutes-per-model'];
    const minutesPerModel = perModelFlag === undefined ? undefined : Number(perModelFlag);
    if (minutesPerModel !== undefined && !(minutesPerModel > 0)) {
      usage(`--minutes-per-model must be a number > 0, got '${String(perModelFlag)}'`);
    }
    try {
      printPlan(opts, minutesPerModel);
    } catch (err) {
      if (err instanceof CatalogError) {
        usage(err.message);
      }
      throw err;
    }
    return;
  }

  const cliPath = fileURLToPath(new URL('./cli.ts', import.meta.url));
  const run: HarnessRunner = (modelIds) => {
    const result = spawnSync(
      process.execPath,
      [
        '--experimental-strip-types',
        cliPath,
        '--catalog',
        catalogPath,
        '--out',
        outDir,
        '--only',
        modelIds.join(','),
        ...positionals,
      ],
      { stdio: 'inherit' },
    );
    return result.status;
  };

  let outcome: BatchOutcome;
  try {
    outcome = runBatches(opts, run);
  } catch (err) {
    if (err instanceof CatalogError) {
      usage(err.message);
    }
    throw err;
  }
  process.stdout.write(
    `batches: ${String(outcome.measured.length)} model(s) with results, ` +
      `${String(outcome.unmeasured.length)} unmeasured, ` +
      `${String(outcome.deferred.length)} deferred to the next run, ` +
      `${String(outcome.invalidRuns)} run(s) with schema-invalid results\n`,
  );
  for (const id of outcome.unmeasured) {
    process.stdout.write(`batches: UNMEASURED ${id}\n`);
  }
  for (const id of outcome.deferred) {
    process.stdout.write(`batches: DEFERRED ${id}\n`);
  }
  // A deferred model is not a failure: it leads the next run (--history).
  process.exitCode = outcome.unmeasured.length > 0 || outcome.invalidRuns > 0 ? 1 : 0;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
