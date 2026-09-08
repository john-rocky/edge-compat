/**
 * Demo-zoo static site generator (Phase 7).
 *
 * `npm run build -- [--out <dir>] [--repo-url <url>] [--cookbook-url <url>]`
 *
 * Reads cards/index.json + per-model card.json (+ linked sweep files),
 * writes a fully static site: index.html (sortable compatibility table,
 * passing AND failing statuses), one demo page per passing model, verbatim
 * card copies, bundled page scripts, and the @litertjs/core WASM assets.
 * Model weights are never copied — demo pages fetch them from their official
 * source URL at runtime.
 *
 * Deterministic: same inputs and dependency versions => byte-identical
 * output. Exit codes: 0 built, 2 usage or input error.
 */
import * as fs from 'node:fs';
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { build } from 'esbuild';

import { PROJECT_NAME } from '../shared/branding.ts';
import type { DemoConfig, SweepDisplayRecord } from '../shared/config.ts';
import type { PlaygroundConfig } from '../shared/playground.ts';
import { type LatestKnownRelease, staleAgainst } from '../shared/staleness.ts';
import { type ModelView, SiteDataError, loadSiteData, resolveModelUrl } from './data.ts';
import { loadWebMatrixSnapshots } from './matrix.ts';
import { renderDemoPage, renderIndexPage, renderPlaygroundPage } from './render.ts';

const require = createRequire(import.meta.url);

const SITE_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const WASM_DIR = path.join(path.dirname(require.resolve('@litertjs/core/package.json')), 'wasm');

const USAGE = `usage: npm run build -- [options]

options:
  --out <dir>            output directory (default: dist, relative to site/)
  --repo-url <url>       repository link for demo-page CTAs (default: omitted —
                         CTA targets are set at build time)
  --cookbook-url <url>   litert-cookbook link for demo-page CTAs (default: omitted)
  --matrix-dir <dir>     matrix snapshot directory whose WEB-backend snapshots are
                         exported for the playground (default: data/matrix,
                         relative to the repo root)
  --releases <path>      latest-known-release registry (default: data/releases.json,
                         relative to the repo root; skipped when absent). Rows and
                         verdicts whose verified-against runtime version (@litertjs/core,
                         litert, or litert-lm) lags the registry get a visible stale
                         flag — display only
  --lineup <file>        curated launch lineup (Phase 12.2): a text file of model
                         ids, one per line ('#' comments allowed). Demo pages are
                         built ONLY for the listed ids; the index table still shows
                         every model, passing and failing. Unknown ids and ids that
                         fail the demo honesty gate are build errors. Default:
                         all eligible models (build unchanged)
`;

/** Backends the playground runs live, reference (wasm_xnnpack) first. */
const PLAYGROUND_RUN_BACKENDS = ['wasm_xnnpack', 'webgpu_mldrift'];

/** Sweep-mirroring playground run parameters (web/sweep defaults, DECISIONS #95). */
const PLAYGROUND_RUN_CONFIG = {
  warmupRuns: 3,
  timedRuns: 10,
  inputSeed: 42,
  toleranceAbs: 1e-5,
  toleranceRel: 1e-3,
};

function fail(message: string): never {
  process.stderr.write(`error: ${message}\n\n${USAGE}`);
  process.exit(2);
}

/**
 * The latest known releases from data/releases.json (Phase 11; Phase 13 adds
 * the litert and litertlm axes for device-run staleness). An explicit
 * --releases path must load; the default path is used only when the file
 * exists. A malformed registry aborts the build — silently building without
 * staleness would hide decay.
 */
function loadKnownReleases(
  explicitPath: string | null,
): Record<string, LatestKnownRelease> {
  const releasesPath = explicitPath ?? path.join(REPO_ROOT, 'data', 'releases.json');
  if (explicitPath === null && !fs.existsSync(releasesPath)) {
    return {};
  }
  let doc: unknown;
  try {
    doc = JSON.parse(fs.readFileSync(releasesPath, 'utf8'));
  } catch (err) {
    throw new SiteDataError(`${releasesPath}: cannot read releases registry: ${String(err)}`);
  }
  const releases = (doc as { releases?: Record<string, unknown> }).releases ?? {};
  const known: Record<string, LatestKnownRelease> = {};
  for (const [axis, value] of Object.entries(releases)) {
    const entry = value as { version?: unknown; checked_at?: unknown };
    if (typeof entry.version !== 'string' || typeof entry.checked_at !== 'string') {
      throw new SiteDataError(
        `${releasesPath}: releases.${axis} must carry string 'version' and 'checked_at'`,
      );
    }
    known[axis] = { version: entry.version, checkedAt: entry.checked_at };
  }
  return known;
}

function ctaPrompt(modelId: string, modelUrl: string, license: string): string {
  return (
    `Use the litert-cookbook web skill to build a LiteRT.js browser demo for ` +
    `${modelId}: load the model from ${modelUrl} (license: ${license}) and run it ` +
    `with @litertjs/core.`
  );
}

/**
 * The demo page's "reproduce this with an agent" prompt (Phase 12.2): a
 * one-row catalog for this exact model, then the real sweep command.
 */
function demoReproPrompt(model: ModelView, modelUrl: string): string {
  const inputSpec = model
    .demo!.inputs.map((tensor) => `${tensor.shape.join('x')}:${tensor.dtype}`)
    .join(';');
  return (
    `In the ${PROJECT_NAME} repo, verify ${model.id} with the sweep harness: write ` +
    '/tmp/catalog.csv with the header line model_id,source,input_spec,license,notes ' +
    `and the single row ${model.id},${modelUrl},${inputSpec},${model.license}, — ` +
    'then run: cd web/sweep && npm ci && npm run sweep -- --catalog /tmp/catalog.csv ' +
    `--out out/ — and compare out/${model.id}.json with the sweep table on this page.`
  );
}

function demoConfigFor(
  model: ModelView,
  litertjsVersion: string,
  repoUrl: string | null,
  cookbookUrl: string | null,
  latestKnown: LatestKnownRelease | null,
): DemoConfig {
  const demo = model.demo!;
  const browser = model.browser!;
  const modelUrl = resolveModelUrl(model.sourceUrl, model.artifactFile!);
  const sweep: SweepDisplayRecord[] = browser.records.map((record) => ({
    backend: record.backend,
    status: browser.statuses[record.backend]!,
    fullDelegation: record.fullDelegation,
    outputMatch: record.outputMatch,
    latencyP50Ms: record.latencyP50Ms,
    envLabel: record.envLabel,
    date: record.date,
    provenance: record.provenance,
    coreVersion: record.coreVersion,
    staleAgainst: staleAgainst(record.coreVersion, latestKnown),
  }));
  return {
    modelId: model.id,
    modelUrl,
    license: model.license,
    sourceUrl: model.sourceUrl,
    inputs: demo.inputs,
    inputSeed: demo.inputSeed,
    warmupRuns: demo.warmupRuns,
    timedRuns: demo.timedRuns,
    sweep,
    wasmBaseUrl: '../../assets/wasm/',
    litertjsVersion,
    ctaPrompt: ctaPrompt(model.id, modelUrl, model.license),
    reproPrompt: demoReproPrompt(model, modelUrl),
    repoUrl,
    cookbookUrl,
  };
}

async function bundlePageScripts(outDir: string): Promise<void> {
  await build({
    entryPoints: [
      path.join(SITE_ROOT, 'src', 'page', 'demo.ts'),
      path.join(SITE_ROOT, 'src', 'page', 'sort.ts'),
      path.join(SITE_ROOT, 'src', 'page', 'playground.ts'),
      path.join(SITE_ROOT, 'src', 'page', 'copy.ts'),
    ],
    bundle: true,
    format: 'iife',
    outdir: path.join(outDir, 'assets'),
    entryNames: '[name]',
    logLevel: 'silent',
  });
}

function copyWasmAssets(outDir: string): void {
  const target = path.join(outDir, 'assets', 'wasm');
  fs.mkdirSync(target, { recursive: true });
  for (const name of fs.readdirSync(WASM_DIR).sort()) {
    fs.copyFileSync(path.join(WASM_DIR, name), path.join(target, name));
  }
}

async function main(): Promise<void> {
  let values;
  try {
    ({ values } = parseArgs({
      options: {
        out: { type: 'string', default: 'dist' },
        'repo-url': { type: 'string' },
        'cookbook-url': { type: 'string' },
        'matrix-dir': { type: 'string', default: path.join('data', 'matrix') },
        releases: { type: 'string' },
        lineup: { type: 'string' },
      },
      strict: true,
    }));
  } catch (err) {
    fail(err instanceof Error ? err.message : String(err));
  }
  const outDir = path.resolve(SITE_ROOT, values.out);
  if (outDir === REPO_ROOT || REPO_ROOT.startsWith(outDir + path.sep)) {
    fail(`--out ${values.out} would delete the repository`);
  }
  const repoUrl = values['repo-url'] ?? null;
  const cookbookUrl = values['cookbook-url'] ?? null;
  const matrixDir = path.resolve(REPO_ROOT, values['matrix-dir']);
  const knownReleases = loadKnownReleases(
    values.releases !== undefined ? path.resolve(REPO_ROOT, values.releases) : null,
  );
  const latestKnown = knownReleases['litertjs_core'] ?? null;
  const deviceReleases = {
    litert: knownReleases['litert'] ?? null,
    litertlm: knownReleases['litertlm'] ?? null,
  };

  const models = loadSiteData(REPO_ROOT);

  // Curated launch lineup (Phase 12.2): demo pages only for the listed ids;
  // the index table is untouched — passing AND failing models stay listed.
  let lineupSize: number | null = null;
  if (values.lineup !== undefined) {
    const lineupPath = path.resolve(REPO_ROOT, values.lineup);
    let text: string;
    try {
      text = fs.readFileSync(lineupPath, 'utf8');
    } catch (err) {
      fail(`--lineup ${values.lineup}: ${String(err)}`);
    }
    const ids = text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line !== '' && !line.startsWith('#'));
    if (ids.length === 0) {
      fail(`--lineup ${values.lineup}: no model ids listed`);
    }
    const byId = new Map(models.map((model) => [model.id, model]));
    const problems: string[] = [];
    for (const id of ids) {
      const model = byId.get(id);
      if (model === undefined) {
        problems.push(`lineup id '${id}' is not in cards/index.json`);
      } else if (model.demo === null) {
        problems.push(`lineup id '${id}' cannot have a demo page: ${model.demoSkipReason ?? 'unknown'}`);
      }
    }
    if (problems.length > 0) {
      fail(`--lineup ${values.lineup}:\n  ${problems.join('\n  ')}`);
    }
    const lineup = new Set(ids);
    for (const model of models) {
      if (model.demo !== null && !lineup.has(model.id)) {
        model.demo = null;
        model.demoSkipReason = 'not in the launch lineup (--lineup)';
      }
    }
    lineupSize = ids.length;
  }

  const matrixSnapshots = loadWebMatrixSnapshots(matrixDir);
  const corePkg = require('@litertjs/core/package.json') as { version: string };

  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  // GitHub Pages: serve as-is, no Jekyll pass.
  fs.writeFileSync(path.join(outDir, '.nojekyll'), '');

  await bundlePageScripts(outDir);
  copyWasmAssets(outDir);
  fs.copyFileSync(
    path.join(SITE_ROOT, 'static', 'site.css'),
    path.join(outDir, 'assets', 'site.css'),
  );
  // Discovery entry points for agents at the site root (RELEASE.md step 7):
  // the committed llms.txt and the card index, verbatim.
  fs.copyFileSync(path.join(REPO_ROOT, 'llms.txt'), path.join(outDir, 'llms.txt'));
  fs.mkdirSync(path.join(outDir, 'cards'), { recursive: true });
  fs.copyFileSync(
    path.join(REPO_ROOT, 'cards', 'index.json'),
    path.join(outDir, 'cards', 'index.json'),
  );

  let demoPages = 0;
  const skipped: string[] = [];
  for (const model of models) {
    const cardDir = path.join(outDir, 'cards', model.id);
    fs.mkdirSync(cardDir, { recursive: true });
    fs.copyFileSync(model.cardJsonAbs, path.join(cardDir, 'card.json'));
    fs.copyFileSync(model.cardMdAbs, path.join(cardDir, 'CARD.md'));
    if (model.sweepAbs !== null) {
      fs.mkdirSync(path.join(outDir, 'sweeps'), { recursive: true });
      fs.copyFileSync(model.sweepAbs, path.join(outDir, 'sweeps', `${model.id}.json`));
    }
    if (model.demo !== null) {
      const config = demoConfigFor(model, corePkg.version, repoUrl, cookbookUrl, latestKnown);
      const pageDir = path.join(outDir, 'models', model.id);
      fs.mkdirSync(pageDir, { recursive: true });
      fs.writeFileSync(path.join(pageDir, 'index.html'), renderDemoPage(model, config));
      demoPages += 1;
    } else {
      skipped.push(`  demo skipped: ${model.id} — ${model.demoSkipReason ?? 'unknown'}`);
    }
  }
  fs.writeFileSync(
    path.join(outDir, 'index.html'),
    renderIndexPage(models, latestKnown, deviceReleases),
  );

  // Playground (Phase 9): verbatim web-backend matrix snapshot copies + the
  // check page. Zero snapshots is the honest default until real web-backend
  // matrix data exists — the page then renders every op as 'unknown'.
  if (matrixSnapshots.length > 0) {
    fs.mkdirSync(path.join(outDir, 'matrix'), { recursive: true });
    for (const snapshot of matrixSnapshots) {
      fs.copyFileSync(snapshot.abs, path.join(outDir, 'matrix', `${snapshot.backend}.json`));
    }
  }
  // The playground's "reproduce this with an agent" prompt (Phase 12.2):
  // the real edge-lint command against a real committed matrix file — the
  // first exported web snapshot, or honestly the example-provenance matrix
  // when the default build exports none.
  const lintMatrix =
    matrixSnapshots.length > 0
      ? `data/matrix/${path.basename(matrixSnapshots[0]!.abs)}`
      : 'data/examples/matrix_example.json';
  const playgroundRepro =
    `In the ${PROJECT_NAME} repo, reproduce this page's static verdicts for my model ` +
    'locally — the model stays on my machine: uv sync && uv run edge-lint ' +
    `MODEL.tflite --matrix ${lintMatrix} --json` +
    (matrixSnapshots.length === 0
      ? ' (no web-backend snapshots are exported yet, so this names the example-provenance matrix).'
      : '.');

  const playgroundConfig: PlaygroundConfig = {
    backends: PLAYGROUND_RUN_BACKENDS,
    reproPrompt: playgroundRepro,
    matrix: matrixSnapshots.map((snapshot) => ({
      backend: snapshot.backend,
      litertVersion: snapshot.litertVersion,
      generatedAt: snapshot.generatedAt,
      entryCount: snapshot.entryCount,
      href: `../matrix/${snapshot.backend}.json`,
      staleAgainst: staleAgainst(snapshot.litertVersion, latestKnown),
    })),
    ...PLAYGROUND_RUN_CONFIG,
    wasmBaseUrl: '../assets/wasm/',
    litertjsVersion: corePkg.version,
    repoUrl,
    cookbookUrl,
  };
  fs.mkdirSync(path.join(outDir, 'check'), { recursive: true });
  fs.writeFileSync(
    path.join(outDir, 'check', 'index.html'),
    renderPlaygroundPage(playgroundConfig),
  );

  process.stdout.write(
    `site: ${String(models.length)} model(s) · ${String(demoPages)} demo page(s)` +
      (lineupSize === null ? '' : ` (lineup: ${String(lineupSize)} id(s))`) +
      ` · playground with ${String(matrixSnapshots.length)} matrix snapshot(s) -> ${outDir}\n`,
  );
  for (const line of skipped) {
    process.stdout.write(`${line}\n`);
  }
}

main().catch((err: unknown) => {
  if (err instanceof SiteDataError) {
    fail(err.message);
  }
  process.stderr.write(`error: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  process.exit(2);
});
