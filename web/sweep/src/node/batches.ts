/**
 * Batch driver for a full-catalog sweep: runs the harness in a FRESH process
 * per `--batch` models, because the runner accumulates memory across the
 * models of one run (README "Traps": a 36-model batch died at 27 with the
 * heap exhausted; the same models one at a time all landed). A batch whose
 * process dies is re-run one model at a time; a model that kills a process
 * on its own is reported as UNMEASURED and the driver exits 1 — the output
 * directory still holds every model that was measured, and a re-run skips
 * them. This is what the weekly CI resweep runs (DECISIONS #166).
 *
 * usage: node --experimental-strip-types src/node/batches.ts \
 *          --catalog <csv> --out <dir> [--batch <n>] [-- <harness flags...>]
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
}

export interface BatchOutcome {
  /** Models with a result file in `outDir` when the driver finished, catalog order. */
  measured: string[];
  /** Models still without a result after their batch run and a run alone. */
  unmeasured: string[];
  /** Harness runs that exited 1: a result file failed schema validation. */
  invalidRuns: number;
}

/** Runs the harness on exactly these models; returns its exit status (null = killed by a signal). */
export type HarnessRunner = (modelIds: readonly string[]) => number | null;

export function runBatches(
  opts: BatchOptions,
  run: HarnessRunner,
  log: (line: string) => void = (line) => process.stdout.write(`${line}\n`),
): BatchOutcome {
  const catalogDir = path.dirname(path.resolve(opts.catalogPath));
  const entries = parseCatalog(fs.readFileSync(opts.catalogPath, 'utf8'), catalogDir);
  const done = (id: string): boolean => fs.existsSync(path.join(opts.outDir, `${id}.json`));
  const all = entries.map((entry) => entry.modelId);
  const pending = all.filter((id) => !done(id));
  if (pending.length < all.length) {
    log(
      `batches: ${String(all.length - pending.length)} model(s) already have a result in ` +
        `${opts.outDir} — skipped`,
    );
  }
  const unmeasured: string[] = [];
  let invalidRuns = 0;
  const note = (status: number | null): void => {
    if (status === 1) {
      invalidRuns += 1;
    }
  };
  const describe = (status: number | null): string =>
    status === null ? 'by signal' : `with status ${String(status)}`;

  for (let start = 0; start < pending.length; start += opts.batchSize) {
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
      const single = run([id]);
      note(single);
      if (!done(id)) {
        unmeasured.push(id);
        log(`batches: UNMEASURED ${id} — the harness exited ${describe(single)} on it alone`);
      }
    }
  }
  return { measured: all.filter(done), unmeasured, invalidRuns };
}

const USAGE =
  'usage: node --experimental-strip-types src/node/batches.ts --catalog <csv> --out <dir> ' +
  '[--batch <n>] [-- <harness flags...>]';

function usage(message: string): never {
  process.stderr.write(`error: ${message}\n\n${USAGE}\n`);
  process.exit(2);
}

function main(): void {
  let parsed;
  try {
    parsed = parseArgs({
      options: {
        catalog: { type: 'string' },
        out: { type: 'string' },
        batch: { type: 'string', default: '8' },
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
  const batchSize = Number.parseInt(values.batch, 10);
  if (!Number.isFinite(batchSize) || batchSize < 1 || String(batchSize) !== values.batch.trim()) {
    usage(`--batch must be an integer >= 1, got '${values.batch}'`);
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
    outcome = runBatches({ catalogPath, outDir, batchSize }, run);
  } catch (err) {
    if (err instanceof CatalogError) {
      usage(err.message);
    }
    throw err;
  }
  process.stdout.write(
    `batches: ${String(outcome.measured.length)} model(s) with results, ` +
      `${String(outcome.unmeasured.length)} unmeasured, ` +
      `${String(outcome.invalidRuns)} run(s) with schema-invalid results\n`,
  );
  for (const id of outcome.unmeasured) {
    process.stdout.write(`batches: UNMEASURED ${id}\n`);
  }
  process.exitCode = outcome.unmeasured.length > 0 || outcome.invalidRuns > 0 ? 1 : 0;
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
