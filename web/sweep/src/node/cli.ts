/**
 * CLI entry point: `npm run sweep -- --catalog <csv> --out <dir> [...]`.
 *
 * Exit codes (repo convention): 0 all result files valid; 1 at least one
 * result file failed schema validation; 2 usage / manifest error.
 */
import { parseArgs } from 'node:util';

import { CatalogError } from './catalog.ts';
import { runSweep } from './sweep.ts';

const USAGE = `usage: npm run sweep -- --catalog <csv> --out <dir> [options]

options:
  --catalog <path>        catalog manifest CSV (required)
  --out <dir>             output directory, one <model_id>.json per model (required)
  --experimental          include the webnn backend
  --headed                run a headed (non-headless) browser
  --machine-label <s>     env.machine_label for every record (default: unlabeled)
  --date <YYYY-MM-DD>     record date (default: today)
  --provenance <p>        measured | example (default: measured)
  --warmup <n>            warmup runs per backend (default: 3)
  --runs <n>              timed runs per backend (default: 10)
  --tolerance-abs <x>     output-match absolute tolerance (default: 1e-5)
  --tolerance-rel <x>     output-match relative tolerance (default: 1e-3)
  --seed <n>              fixture input PRNG seed (default: 42)
  --timeout-ms <n>        per-step timeout per model x backend (default: 60000)
  --cache-dir <dir>       cache URL-sourced model downloads here; a re-run of an
                          unchanged catalog downloads nothing (Phase 11 re-sweep)
  --cache-max-bytes <n>   with --cache-dir: evict least-recently-used models so the
                          cache never exceeds n bytes; 0 keeps nothing between
                          models (CI: the catalog does not fit a runner's disk)
  --only <id,id,...>      sweep only these catalog model_ids
  --skip-existing         skip models whose <out>/<model_id>.json already exists
`;

/**
 * The sweep closes its server and browser in `finally`; should anything else
 * ever keep the event loop alive after the run has ended, leave within 10 s
 * rather than until a CI job timeout (2026-08-27: five hours). unref() keeps a
 * healthy exit as fast as before.
 */
function exitSoon(code: number): void {
  process.exitCode = code;
  setTimeout(() => process.exit(code), 10_000).unref();
}

function fail(message: string): never {
  process.stderr.write(`error: ${message}\n\n${USAGE}`);
  process.exit(2);
}

function parseIntFlag(value: string, flag: string, minimum: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || String(parsed) !== value.trim() || parsed < minimum) {
    fail(`--${flag} must be an integer >= ${String(minimum)}, got '${value}'`);
  }
  return parsed;
}

function parseFloatFlag(value: string, flag: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    fail(`--${flag} must be a non-negative number, got '${value}'`);
  }
  return parsed;
}

function main(): void {
  let values;
  try {
    ({ values } = parseArgs({
      options: {
        catalog: { type: 'string' },
        out: { type: 'string' },
        experimental: { type: 'boolean', default: false },
        headed: { type: 'boolean', default: false },
        'machine-label': { type: 'string', default: 'unlabeled' },
        date: { type: 'string' },
        provenance: { type: 'string', default: 'measured' },
        warmup: { type: 'string', default: '3' },
        runs: { type: 'string', default: '10' },
        'tolerance-abs': { type: 'string', default: '1e-5' },
        'tolerance-rel': { type: 'string', default: '1e-3' },
        seed: { type: 'string', default: '42' },
        'timeout-ms': { type: 'string', default: '60000' },
        'cache-dir': { type: 'string' },
        'cache-max-bytes': { type: 'string' },
        only: { type: 'string' },
        'skip-existing': { type: 'boolean', default: false },
      },
      strict: true,
    }));
  } catch (err) {
    fail(err instanceof Error ? err.message : String(err));
  }

  if (values.catalog === undefined) {
    fail('--catalog is required');
  }
  if (values.out === undefined) {
    fail('--out is required');
  }
  const date = values.date ?? new Date().toISOString().slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    fail(`--date must be YYYY-MM-DD, got '${date}'`);
  }
  const provenance = values.provenance;
  if (provenance !== 'measured' && provenance !== 'example') {
    fail(`--provenance must be 'measured' or 'example', got '${provenance}'`);
  }
  if (values['cache-max-bytes'] !== undefined && values['cache-dir'] === undefined) {
    fail('--cache-max-bytes needs --cache-dir');
  }
  const only =
    values.only === undefined
      ? undefined
      : values.only
          .split(',')
          .map((id) => id.trim())
          .filter((id) => id.length > 0);
  if (only !== undefined && only.length === 0) {
    fail('--only needs at least one model_id');
  }

  runSweep({
    catalogPath: values.catalog,
    outDir: values.out,
    experimental: values.experimental,
    headed: values.headed,
    machineLabel: values['machine-label'],
    date,
    provenance,
    cacheDir: values['cache-dir'],
    cacheMaxBytes:
      values['cache-max-bytes'] === undefined
        ? undefined
        : parseIntFlag(values['cache-max-bytes'], 'cache-max-bytes', 0),
    only,
    skipExisting: values['skip-existing'],
    timeoutMs: parseIntFlag(values['timeout-ms'], 'timeout-ms', 1000),
    config: {
      warmup_runs: parseIntFlag(values.warmup, 'warmup', 0),
      timed_runs: parseIntFlag(values.runs, 'runs', 1),
      tolerance_abs: parseFloatFlag(values['tolerance-abs'], 'tolerance-abs'),
      tolerance_rel: parseFloatFlag(values['tolerance-rel'], 'tolerance-rel'),
      input_seed: parseIntFlag(values.seed, 'seed', 0),
    },
  }).then(
    (summary) => {
      exitSoon(summary.exitCode);
    },
    (err: unknown) => {
      const errnoCode = (err as NodeJS.ErrnoException | null)?.code;
      if (err instanceof CatalogError || (err instanceof Error && errnoCode === 'ENOENT')) {
        fail(err.message);
      }
      process.stderr.write(`error: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
      exitSoon(2);
    },
  );
}

main();
