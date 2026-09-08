/**
 * Generator tests: pure units (status vocabulary, honesty gate, URL/spec
 * parsing, failing-status rendering) plus full-build assertions —
 * byte-determinism across two runs, verbatim card copies, demo pages only
 * for passing models, and the weights-in-repo check (no .tflite in output).
 */
import * as assert from 'node:assert/strict';
import { type SpawnSyncReturns, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  type CardBrowserRecord,
  type CardDeviceRecord,
  type ModelView,
  browserStatus,
  deviceStatus,
  isDemoEligible,
  loadSiteData,
  parseShapeSpec,
  resolveModelUrl,
} from '../src/generator/data.ts';
import { loadWebMatrixSnapshots } from '../src/generator/matrix.ts';
import { INDEX_REPRODUCE_PROMPT, escapeHtml, renderIndexPage } from '../src/generator/render.ts';
import { DISCLOSURE_LINE, NAMING_NOTICE } from '../src/shared/branding.ts';
import { lagsBehind, parseVersion, staleAgainst } from '../src/shared/staleness.ts';

const SITE_ROOT = fileURLToPath(new URL('..', import.meta.url));
const REPO_ROOT = path.resolve(SITE_ROOT, '..');

function record(overrides: Partial<CardBrowserRecord>): CardBrowserRecord {
  return {
    backend: 'webgpu_mldrift',
    date: '2026-08-10',
    fullDelegation: true,
    latencyP50Ms: 1.0,
    loads: true,
    maxRelDiff: null,
    outputMatch: true,
    provenance: 'example',
    runs: true,
    envLabel: 'test-env · chromium 1.0 headless · macOS · @litertjs/core 0.0.0',
    coreVersion: '0.0.0',
    ...overrides,
  };
}

function runBuildRaw(outDir: string, extraArgs: string[] = []): SpawnSyncReturns<string> {
  return spawnSync(
    process.execPath,
    [
      '--experimental-strip-types',
      path.join('src', 'generator', 'cli.ts'),
      '--out',
      outDir,
      ...extraArgs,
    ],
    { cwd: SITE_ROOT, encoding: 'utf8' },
  );
}

function runBuild(outDir: string, extraArgs: string[] = []): void {
  const result = runBuildRaw(outDir, extraArgs);
  assert.equal(result.status, 0, `build failed:\n${result.stdout}\n${result.stderr}`);
}

function walk(dir: string, prefix = ''): string[] {
  const files: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
    a.name < b.name ? -1 : 1,
  )) {
    const rel = path.join(prefix, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(path.join(dir, entry.name), rel));
    } else {
      files.push(rel);
    }
  }
  return files;
}

void describe('status vocabulary (mirrors litert_compat.cards.index, DECISIONS #62)', () => {
  void test('all five words derive deterministically', () => {
    assert.equal(browserStatus(record({ loads: false, runs: false })), 'load_failed');
    assert.equal(browserStatus(record({ runs: false })), 'run_failed');
    assert.equal(browserStatus(record({ outputMatch: false })), 'output_mismatch');
    assert.equal(browserStatus(record({ fullDelegation: false })), 'fallback');
    assert.equal(browserStatus(record({})), 'pass');
    // null full_delegation / output_match (e.g. the wasm reference) never demotes.
    assert.equal(browserStatus(record({ fullDelegation: null, outputMatch: null })), 'pass');
  });
});

void describe('honesty gate', () => {
  void test('no demo without at least one backend that ran', () => {
    assert.equal(isDemoEligible([record({ loads: false, runs: false })]), false);
    assert.equal(isDemoEligible([record({ runs: false })]), false);
    assert.equal(isDemoEligible([]), false);
    assert.equal(isDemoEligible([record({ runs: false }), record({})]), true);
  });
});

void describe('spec/url parsing', () => {
  void test('inline shape specs parse, fixture files and garbage do not', () => {
    assert.deepEqual(parseShapeSpec('1x64:float32'), [{ shape: [1, 64], dtype: 'float32' }]);
    assert.deepEqual(parseShapeSpec('1x8:float32;4:int32'), [
      { shape: [1, 8], dtype: 'float32' },
      { shape: [4], dtype: 'int32' },
    ]);
    assert.equal(parseShapeSpec('inputs.json'), null);
    assert.equal(parseShapeSpec('1x64:float16'), null);
  });

  void test('model URL: .tflite passthrough, otherwise HF-style resolve', () => {
    assert.equal(resolveModelUrl('https://x.test/m.tflite', 'ignored.tflite'), 'https://x.test/m.tflite');
    assert.equal(
      resolveModelUrl('https://example.invalid/models/example-web-a', 'model_web_a_example.tflite'),
      'https://example.invalid/models/example-web-a/resolve/main/model_web_a_example.tflite',
    );
  });
});

void describe('staleness (Phase 11, mirrors litert_compat.freshness.releases)', () => {
  void test('numeric-prefix version comparison, strictly-numeric guard', () => {
    assert.deepEqual(parseVersion('2.5.3'), [2, 5, 3]);
    assert.deepEqual(parseVersion('0.1.0-rc1'), [0, 1, 0]);
    assert.equal(parseVersion('nightly'), null);
    assert.equal(lagsBehind('2.5.2', '2.5.3'), true);
    assert.equal(lagsBehind('2.5', '2.5.3'), true);
    assert.equal(lagsBehind('2.5.3', '2.5.3'), false);
    assert.equal(lagsBehind('2.5.4', '2.5.3'), false);
    // Placeholder versions make no claim about a real runtime: never stale.
    assert.equal(lagsBehind('0.0.0-example', '2.5.3'), false);
    assert.equal(lagsBehind('nightly', '2.5.3'), false);
  });

  void test('staleAgainst returns the release only when actually lagging', () => {
    const latest = { version: '2.5.3', checkedAt: '2026-08-10' };
    assert.deepEqual(staleAgainst('2.5.2', latest), latest);
    assert.equal(staleAgainst('2.5.3', latest), null);
    assert.equal(staleAgainst('2.5.2', null), null);
  });

  void test('index rows carry the stale flag exactly when the record lags', () => {
    const models = loadSiteData(REPO_ROOT); // committed records: core 2.5.3
    const stale = renderIndexPage(models, { version: '9.9.9', checkedAt: '2026-08-10' });
    assert.match(stale, /stale-flag/);
    assert.match(stale, /stale — latest known 9\.9\.9/);
    const current = renderIndexPage(models, { version: '2.5.3', checkedAt: '2026-08-10' });
    assert.doesNotMatch(current, /stale-flag/);
    const noRegistry = renderIndexPage(models);
    assert.doesNotMatch(noRegistry, /stale-flag/);
    // The flag never changes the status or measurement it sits next to.
    const strip = (html: string): string =>
      html.replaceAll(/ <span class="stale-flag".*?<\/span>/g, '');
    assert.equal(strip(stale), strip(current));
  });
});

function deviceRecord(overrides: Partial<CardDeviceRecord>): CardDeviceRecord {
  return {
    device: 'test-phone',
    accelerator: 'npu_qnn',
    date: '2026-08-11',
    loads: true,
    runs: true,
    outputMatch: null,
    fullDelegation: null,
    latencyP50Ms: null,
    decodeTokensPerS: null,
    provenance: 'example',
    runtime: 'litert',
    runtimeVersion: '0.0.0-example',
    envLabel: 'Test Phone · litert 0.0.0-example',
    ...overrides,
  };
}

void describe('device runs column (Phase 13)', () => {
  void test('status vocabulary mirrors litert_compat.device_runs.records', () => {
    assert.equal(deviceStatus(deviceRecord({})), 'pass');
    assert.equal(deviceStatus(deviceRecord({ loads: false, runs: false })), 'load_failed');
    assert.equal(deviceStatus(deviceRecord({ runs: false })), 'run_failed');
    assert.equal(deviceStatus(deviceRecord({ outputMatch: false })), 'output_mismatch');
    assert.equal(deviceStatus(deviceRecord({ fullDelegation: false })), 'fallback');
  });

  void test('committed device records render env-labeled in the index table', () => {
    const models = loadSiteData(REPO_ROOT);
    const html = renderIndexPage(models);
    assert.match(html, /Device runs/);
    // The committed example NPU record is a fallback with its env label.
    assert.match(html, /example-phone<\/code> npu_qnn:/);
    assert.match(html, /status-fallback/);
    assert.match(html, /Example Phone · Example SoC · litert 0\.0\.0-example/);
    // The LLM-lane record shows decode throughput and the litert-lm env.
    assert.match(html, /45\.6 tok\/s decode/);
    assert.match(html, /litert-lm 0\.0\.0-example/);
  });

  void test('device stale flag follows the record runtime axis', () => {
    const models = loadSiteData(REPO_ROOT);
    const withDevice: ModelView = {
      ...models[0]!,
      id: 'synthetic-device-model',
      device: {
        records: [
          deviceRecord({ runtime: 'litert', runtimeVersion: '2.1.5' }),
          deviceRecord({
            device: 'test-mac',
            accelerator: 'gpu',
            runtime: 'litert-lm',
            runtimeVersion: '0.14.0',
          }),
        ],
      },
    };
    const releases = {
      litert: { version: '2.1.6', checkedAt: '2026-08-11' },
      litertlm: { version: '0.15.0', checkedAt: '2026-08-11' },
    };
    const html = renderIndexPage([...models, withDevice], null, releases);
    assert.match(html, /stale — latest known 2\.1\.6/);
    assert.match(html, /stale — latest known 0\.15\.0/);
    assert.match(html, /older litert than/);
    assert.match(html, /older litert-lm than/);
    // Current versions carry no flag.
    const current = renderIndexPage([...models, withDevice], null, {
      litert: { version: '2.1.5', checkedAt: '2026-08-11' },
      litertlm: { version: '0.14.0', checkedAt: '2026-08-11' },
    });
    assert.doesNotMatch(current, /stale-flag/);
    // No registry, no flag: without a latest-known release nothing is guessed
    // stale (the committed device records are real measurements now).
    const noDeviceRegistry = renderIndexPage(models, null, { litert: null, litertlm: null });
    assert.doesNotMatch(noDeviceRegistry, /stale-flag/);
  });
});

void describe('index rendering', () => {
  void test('passing AND failing statuses render in the same table', () => {
    const models = loadSiteData(REPO_ROOT);
    const failing: ModelView = {
      id: 'synthetic-failing-model',
      family: 'test',
      task: 'test',
      license: 'apache-2.0',
      sourceUrl: 'https://example.invalid/models/synthetic-failing-model',
      delegation: null,
      browser: {
        statuses: { webgpu_mldrift: 'load_failed' },
        sweepSource: 'data/examples/sweep/none.json',
        records: [record({ loads: false, runs: false })],
      },
      device: null,
      artifactFile: null,
      cardJsonAbs: '',
      cardMdAbs: '',
      sweepAbs: null,
      demo: null,
      demoSkipReason: 'no backend with runs=true (honesty gate)',
    };
    const html = renderIndexPage([...models, failing]);
    assert.match(html, /status-pass/);
    assert.match(html, /load_failed/);
    assert.match(html, /not measured/);
    // The failing model is listed, with its status, and has no demo link.
    assert.match(html, /synthetic-failing-model/);
    assert.doesNotMatch(html, /models\/synthetic-failing-model\//);
    // Passing models do link their demo pages.
    assert.match(html, /models\/example-web-a\//);
  });
});

void describe('full build', () => {
  let outA = '';
  let outB = '';

  void before(() => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-site-'));
    outA = path.join(base, 'a');
    outB = path.join(base, 'b');
    runBuild(outA);
    runBuild(outB);
  });

  void after(() => {
    fs.rmSync(path.dirname(outA), { recursive: true, force: true });
  });

  void test('byte-identical across two runs', () => {
    const filesA = walk(outA);
    const filesB = walk(outB);
    assert.deepEqual(filesA, filesB);
    for (const rel of filesA) {
      const a = fs.readFileSync(path.join(outA, rel));
      const b = fs.readFileSync(path.join(outB, rel));
      assert.ok(a.equals(b), `${rel} differs between two builds`);
    }
  });

  void test('demo pages exist exactly for models that pass the honesty gate', () => {
    // Data-driven since real sweep data landed (2026-08-11): the expectation
    // is computed from the same gate the generator applies, so the test pins
    // the INVARIANT (pages ⇔ demo-eligible models), not a model list.
    const demoDirs = fs.readdirSync(path.join(outA, 'models')).sort();
    const expected = loadSiteData(REPO_ROOT)
      .filter((m) => m.demo !== null)
      .map((m) => m.id)
      .sort();
    assert.deepEqual(demoDirs, expected);
    assert.ok(expected.includes('example-web-a'), 'example fixtures stay demo-eligible');
    for (const id of demoDirs) {
      assert.ok(fs.existsSync(path.join(outA, 'models', id, 'index.html')));
    }
  });

  void test('weights-in-repo check: no model binaries in the site output', () => {
    const binaries = walk(outA).filter((f) => f.endsWith('.tflite'));
    assert.deepEqual(binaries, []);
  });

  void test('playground page exports the committed web snapshots (Phase 9)', () => {
    // data/matrix/ carries real web-backend snapshots since 2026-08-11; the
    // default build exports each one and the page lists it with a raw-JSON
    // link. The empty-matrix branch keeps its own test below.
    const snapshots = loadWebMatrixSnapshots(path.join(REPO_ROOT, 'data', 'matrix'));
    assert.ok(snapshots.length >= 2, 'committed web snapshots expected');
    const html = fs.readFileSync(path.join(outA, 'check', 'index.html'), 'utf8');
    assert.ok(!html.includes('No web-backend matrix snapshots are exported yet'));
    assert.match(html, /never leaves your browser/);
    for (const snapshot of snapshots) {
      assert.ok(html.includes(`<code>${snapshot.backend}</code>`), `${snapshot.backend} listed`);
      assert.ok(
        fs.existsSync(path.join(outA, 'matrix', `${snapshot.backend}.json`)),
        `${snapshot.backend} snapshot exported`,
      );
    }
    // The index links the playground.
    const index = fs.readFileSync(path.join(outA, 'index.html'), 'utf8');
    assert.match(index, /href="check\/"/);
  });

  void test('playground stays honest when no web snapshots exist (empty --matrix-dir)', () => {
    const emptyMatrix = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-site-empty-matrix-'));
    const out = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-site-empty-out-'));
    try {
      runBuild(out, ['--matrix-dir', emptyMatrix]);
      const html = fs.readFileSync(path.join(out, 'check', 'index.html'), 'utf8');
      assert.match(html, /No web-backend matrix snapshots are exported yet/);
      assert.ok(!fs.existsSync(path.join(out, 'matrix')), 'no matrix dir when nothing is exported');
      assert.ok(html.includes('example-provenance matrix'));
    } finally {
      fs.rmSync(emptyMatrix, { recursive: true, force: true });
      fs.rmSync(out, { recursive: true, force: true });
    }
  });

  void test('card copies are byte-verbatim', () => {
    for (const id of ['example-tiny-clean', 'example-tiny-mixed', 'example-web-a', 'example-web-b']) {
      for (const name of ['card.json', 'CARD.md']) {
        const src = fs.readFileSync(path.join(REPO_ROOT, 'cards', id, name));
        const copy = fs.readFileSync(path.join(outA, 'cards', id, name));
        assert.ok(src.equals(copy), `cards/${id}/${name} copy differs from source`);
      }
    }
  });

  void test('sweep records are copied byte-verbatim for every sweep-linked model', () => {
    const linked = loadSiteData(REPO_ROOT).filter((m) => m.sweepAbs !== null);
    const sweeps = fs.readdirSync(path.join(outA, 'sweeps')).sort();
    assert.deepEqual(
      sweeps,
      linked.map((m) => `${m.id}.json`).sort(),
    );
    assert.ok(sweeps.includes('example-web-a.json'));
    for (const model of linked) {
      const src = fs.readFileSync(model.sweepAbs!);
      const copy = fs.readFileSync(path.join(outA, 'sweeps', `${model.id}.json`));
      assert.ok(src.equals(copy), `${model.id} sweep copy differs from its source`);
    }
  });

  void test('demo config embeds the sweep-matched inputs and the resolved model URL', () => {
    const html = fs.readFileSync(path.join(outA, 'models', 'example-web-a', 'index.html'), 'utf8');
    const match = /<script type="application\/json" id="demo-config">(.*?)<\/script>/s.exec(html);
    assert.ok(match !== null, 'demo-config block missing');
    const config = JSON.parse(match[1]!) as Record<string, unknown>;
    assert.equal(
      config['modelUrl'],
      'https://example.invalid/models/example-web-a/resolve/main/model_web_a_example.tflite',
    );
    assert.deepEqual(config['inputs'], [{ shape: [1, 64], dtype: 'float32' }]);
    assert.equal(config['inputSeed'], 42);
    assert.equal(config['warmupRuns'], 3);
    assert.equal(config['timedRuns'], 10);
    // Repo/cookbook links default to omitted: the repo is private until clearance.
    assert.equal(config['repoUrl'], null);
    assert.equal(config['cookbookUrl'], null);
  });

  void test('index carries the env label next to every latency', () => {
    const html = fs.readFileSync(path.join(outA, 'index.html'), 'utf8');
    assert.match(html, /example-dev-mac-arm64 · chromium 151\.0\.7922\.34 headless/);
  });

  void test('every page carries the disclosure line and naming notice (Phase 12.1)', () => {
    const pages = walk(outA).filter((f) => f.endsWith('.html'));
    assert.ok(pages.length >= 4, `implausibly few pages: ${pages.join(', ')}`);
    for (const rel of pages) {
      const html = fs.readFileSync(path.join(outA, rel), 'utf8');
      assert.ok(html.includes(DISCLOSURE_LINE), `${rel}: disclosure line missing`);
      assert.ok(html.includes(NAMING_NOTICE), `${rel}: naming notice missing`);
    }
  });

  void test('reproduce-with-an-agent prompts name real commands (Phase 12.2)', () => {
    const index = fs.readFileSync(path.join(outA, 'index.html'), 'utf8');
    // The page HTML-escapes the prompt ('&&' etc.); the copy button copies
    // the rendered textContent, which is the original string again.
    assert.ok(index.includes(escapeHtml(INDEX_REPRODUCE_PROMPT)));
    assert.ok(
      index.includes('npm run sweep -- --catalog ../../data/examples/web_catalog_example.csv'),
    );
    assert.match(index, /<script src="assets\/copy.js" defer><\/script>/);

    const demo = fs.readFileSync(path.join(outA, 'models', 'example-web-a', 'index.html'), 'utf8');
    assert.ok(demo.includes('model_id,source,input_spec,license,notes'));
    assert.ok(demo.includes('npm run sweep -- --catalog /tmp/catalog.csv'));
    assert.ok(demo.includes('out/example-web-a.json'));
    assert.match(demo, /<script src="\.\.\/\.\.\/assets\/copy.js" defer><\/script>/);

    // The default build exports web snapshots, so the playground prompt names
    // the first exported snapshot's committed file (not the example matrix).
    const snapshots = loadWebMatrixSnapshots(path.join(REPO_ROOT, 'data', 'matrix'));
    const first = path.basename(snapshots[0]!.abs);
    const playground = fs.readFileSync(path.join(outA, 'check', 'index.html'), 'utf8');
    assert.ok(
      playground.includes(`edge-lint MODEL.tflite --matrix data/matrix/${first} --json`),
    );
    assert.ok(!playground.includes('example-provenance matrix'));

    assert.ok(fs.existsSync(path.join(outA, 'assets', 'copy.js')));
    for (const page of [index, demo, playground]) {
      assert.match(page, /<button data-copy-target="repro-prompt">Copy prompt<\/button>/);
    }
  });
});

void describe('launch lineup build (--lineup, Phase 12.2)', () => {
  let base = '';

  void before(() => {
    base = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-lineup-'));
  });

  void after(() => {
    fs.rmSync(base, { recursive: true, force: true });
  });

  void test('demo pages only for curated ids; the index still shows everything', () => {
    const lineupFile = path.join(base, 'lineup.txt');
    fs.writeFileSync(lineupFile, '# 初弾 lineup (test)\nexample-web-a\n');
    const outDir = path.join(base, 'curated');
    runBuild(outDir, ['--lineup', lineupFile]);

    assert.deepEqual(fs.readdirSync(path.join(outDir, 'models')).sort(), ['example-web-a']);
    const index = fs.readFileSync(path.join(outDir, 'index.html'), 'utf8');
    // Both models stay in the table; only the curated one links a demo page.
    assert.match(index, /example-web-b/);
    assert.match(index, /models\/example-web-a\//);
    assert.doesNotMatch(index, /models\/example-web-b\//);
    assert.match(index, /not in the launch lineup/);
  });

  void test('unknown and demo-ineligible lineup ids are build errors', () => {
    const unknown = path.join(base, 'unknown.txt');
    fs.writeFileSync(unknown, 'no-such-model\n');
    const resultUnknown = runBuildRaw(path.join(base, 'x'), ['--lineup', unknown]);
    assert.equal(resultUnknown.status, 2);
    assert.match(resultUnknown.stderr, /'no-such-model' is not in cards\/index\.json/);

    const ineligible = path.join(base, 'ineligible.txt');
    fs.writeFileSync(ineligible, 'example-tiny-clean\n');
    const resultIneligible = runBuildRaw(path.join(base, 'y'), ['--lineup', ineligible]);
    assert.equal(resultIneligible.status, 2);
    assert.match(resultIneligible.stderr, /cannot have a demo page/);

    const empty = path.join(base, 'empty.txt');
    fs.writeFileSync(empty, '# nothing\n');
    const resultEmpty = runBuildRaw(path.join(base, 'z'), ['--lineup', empty]);
    assert.equal(resultEmpty.status, 2);
    assert.match(resultEmpty.stderr, /no model ids listed/);
  });
});
