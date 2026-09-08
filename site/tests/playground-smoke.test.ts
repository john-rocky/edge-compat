/**
 * Playground headless smoke (Phase 9 DoD).
 *
 * Builds the site with a matrix dir containing one example-provenance
 * webgpu_mldrift snapshot (3 entries — the known-status path, DECISIONS #91)
 * plus the honest-empty wasm_xnnpack snapshot, serves it Pages-like (no
 * COOP/COEP), and drives the drop → results flow for:
 *   - both example web models on BOTH backends (wasm_xnnpack + webgpu_mldrift),
 *   - the deliberately broken fixture (failure path),
 * asserting on every check that no network request after the model is
 * dropped could carry model-derived bytes (spec 9.1 privacy rule).
 */
import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as os from 'node:os';
import * as path from 'node:path';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { type Browser, type BrowserContext, type Page, chromium } from 'playwright';

const SITE_ROOT = fileURLToPath(new URL('..', import.meta.url));
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const EXAMPLES = path.join(REPO_ROOT, 'data', 'examples');

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
};

function startStaticServer(root: string): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    let filePath = path.join(root, decodeURIComponent(url.pathname));
    if (url.pathname.endsWith('/')) {
      filePath = path.join(filePath, 'index.html');
    }
    if (!filePath.startsWith(root) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath)] ?? 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(filePath).pipe(res);
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        throw new Error('server failed to bind');
      }
      resolve({
        baseUrl: `http://127.0.0.1:${String(address.port)}`,
        close: () =>
          new Promise<void>((res2, rej) => {
            server.close((err) => (err ? rej(err) : res2()));
          }),
      });
    });
  });
}

interface SeenRequest {
  method: string;
  url: string;
  postData: string | null;
}

/**
 * The privacy rule, mechanically: after the model is dropped, every network
 * request must be a bodyless same-origin GET for a static asset with no query
 * string — a shape that cannot carry model bytes or anything derived from
 * them. (Matrix snapshots and most assets load at page init, before any model
 * exists; the LiteRT WASM runtime may lazy-load on first check.)
 */
function assertNoModelBytesLeave(requests: readonly SeenRequest[], baseUrl: string): void {
  for (const request of requests) {
    assert.equal(request.method, 'GET', `non-GET request after drop: ${request.url}`);
    assert.equal(request.postData, null, `request with a body after drop: ${request.url}`);
    assert.ok(request.url.startsWith(baseUrl), `cross-origin request after drop: ${request.url}`);
    const { pathname, search } = new URL(request.url);
    assert.equal(search, '', `query string after drop: ${request.url}`);
    assert.ok(
      pathname.startsWith('/assets/') || pathname === '/favicon.ico',
      `unexpected path after drop: ${request.url}`,
    );
  }
}

async function openPlayground(context: BrowserContext, baseUrl: string): Promise<Page> {
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(String(err)));
  (page as Page & { collectedErrors: string[] }).collectedErrors = errors;
  await page.goto(`${baseUrl}/check/`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.getElementById('pg-status')?.textContent?.startsWith('Ready') ?? false,
  );
  return page;
}

/** Drop a fixture, wait for completion, and return every request seen since. */
async function checkFixture(page: Page, fixture: string): Promise<SeenRequest[]> {
  const requests: SeenRequest[] = [];
  page.on('request', (request) => {
    requests.push({ method: request.method(), url: request.url(), postData: request.postData() });
  });
  await page.setInputFiles('#model-input', path.join(EXAMPLES, fixture));
  await page.waitForSelector('body[data-playground-state="done"]', { timeout: 120000 });
  return requests;
}

async function recordCells(page: Page, backend: string): Promise<string[]> {
  const row = page.locator(`#results-body tr[data-backend="${backend}"]`);
  return (await row.locator('td').allTextContents()).map((text) => text.trim());
}

void describe('playground smoke (headless, Pages-like: no COOP/COEP)', () => {
  let base = '';
  let baseUrl = '';
  let closeServer: (() => Promise<void>) | null = null;
  let browser: Browser | null = null;
  let context: BrowserContext | null = null;

  void before(async () => {
    base = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-pg-smoke-'));
    const matrixDir = path.join(base, 'matrix');
    const outDir = path.join(base, 'dist');
    fs.mkdirSync(matrixDir);
    // Known-status path: example-provenance snapshot with entries (webgpu);
    // honest-empty snapshot (wasm) — unknowns must come from the matrix too.
    fs.copyFileSync(
      path.join(EXAMPLES, 'matrix_webgpu_mldrift_playground_example.json'),
      path.join(matrixDir, 'webgpu_mldrift__2.5.3.json'),
    );
    fs.copyFileSync(
      path.join(EXAMPLES, 'matrix_wasm_xnnpack_example.json'),
      path.join(matrixDir, 'wasm_xnnpack__2.5.3.json'),
    );
    // Staleness surfacing (Phase 11): a registry whose latest known release
    // is ahead of the snapshots' 2.5.3, so the stale flags must render —
    // while every verdict below stays exactly as the snapshot states it.
    const releasesPath = path.join(base, 'releases.json');
    fs.writeFileSync(
      releasesPath,
      JSON.stringify({
        releases: {
          litertjs_core: { version: '9.9.9', checked_at: '2026-08-10', source: 'test' },
        },
      }),
    );
    const result = spawnSync(
      process.execPath,
      [
        '--experimental-strip-types',
        path.join('src', 'generator', 'cli.ts'),
        '--out',
        outDir,
        '--matrix-dir',
        matrixDir,
        '--releases',
        releasesPath,
      ],
      { cwd: SITE_ROOT, encoding: 'utf8' },
    );
    assert.equal(result.status, 0, `build failed:\n${result.stdout}\n${result.stderr}`);

    const server = await startStaticServer(outDir);
    baseUrl = server.baseUrl;
    closeServer = server.close;
    // Same flags as the Phase 5 sweep harness: WebGPU + JSPI enabled.
    const args = ['--enable-unsafe-webgpu', '--enable-features=WebAssemblyJSPI'];
    if (process.platform === 'darwin') {
      args.push('--use-angle=metal');
    }
    browser = await chromium.launch({ headless: true, args });
    context = await browser.newContext();
  });

  void after(async () => {
    await context?.close();
    await browser?.close();
    await closeServer?.();
    fs.rmSync(base, { recursive: true, force: true });
  });

  void test('model A: drop → results on both backends; known + unknown x-ref', async () => {
    const page = await openPlayground(context!, baseUrl);
    const requests = await checkFixture(page, 'model_web_a_example.tflite');
    assertNoModelBytesLeave(requests, baseUrl);

    // Live records: both backends ran, entirely in-browser.
    const wasm = await recordCells(page, 'wasm_xnnpack');
    assert.equal(wasm[1], 'yes', 'wasm loads');
    assert.equal(wasm[2], 'yes', 'wasm runs');
    assert.match(wasm[5]!, /^\d+(\.\d+)?$/, `wasm latency, got '${wasm[5] ?? ''}'`);
    const webgpu = await recordCells(page, 'webgpu_mldrift');
    assert.equal(webgpu[1], 'yes', 'webgpu loads');
    assert.equal(webgpu[2], 'yes', 'webgpu runs');
    assert.equal(webgpu[3], 'yes', 'webgpu fully delegated');
    assert.match(webgpu[4]!, /pass/, `webgpu output match, got '${webgpu[4] ?? ''}'`);
    assert.match(webgpu[5]!, /^\d+(\.\d+)?$/, 'webgpu latency');

    // Env block: the visitor's environment, shown next to the numbers.
    const env = (await page.textContent('#env-block')) ?? '';
    assert.match(env, /Your environment: .+@litertjs\/core/s);

    // Op cross-reference: known status verbatim from the exported matrix
    // (webgpu snapshot has LOGISTIC=delegated with a rewrite hint), unknown
    // everywhere the matrix is silent — including the honest-empty wasm
    // snapshot column.
    const logisticRow = page.locator('#xref-body tr', { hasText: 'LOGISTIC' });
    const logisticCells = await logisticRow.locator('td').allTextContents();
    // Columns: op, nodes, wasm_xnnpack, webgpu_mldrift (sorted).
    assert.match(logisticCells[2]!, /unknown/);
    assert.match(logisticCells[3]!, /delegated/);
    assert.match(logisticCells[3]!, /example rewrite hint for LOGISTIC/);
    assert.match(logisticCells[3]!, /example/);
    const negRow = page.locator('#xref-body tr', { hasText: 'NEG' });
    const negCells = await negRow.locator('td').allTextContents();
    assert.match(negCells[2]!, /unknown/);
    assert.match(negCells[3]!, /unknown/);

    // Staleness surfacing (Phase 11): every matrix verdict column names its
    // verified-against version + date, with a visible stale flag (the build's
    // registry says 9.9.9 is out) — while the verdicts above are unchanged.
    const headers = await page.locator('#xref-head th').allTextContents();
    const webgpuHeader = headers.find((h) => h.includes('webgpu_mldrift')) ?? '';
    assert.match(webgpuHeader, /verified against @litertjs\/core 2\.5\.3 \(2026-08-10\)/);
    assert.match(webgpuHeader, /stale — latest known 9\.9\.9/);
    // The build-time snapshot table carries the same flag.
    const snapshotTable = (await page.textContent('body')) ?? '';
    assert.match(snapshotTable, /stale — latest known 9\.9\.9/);

    // CTA: local lint command names the real snapshot; prompt names the file.
    const lint = (await page.textContent('#lint-command')) ?? '';
    assert.equal(
      lint,
      'edge-lint model_web_a_example.tflite --backend webgpu_mldrift ' +
        '--matrix data/matrix/webgpu_mldrift__2.5.3.json --md',
    );
    const prompt = (await page.textContent('#cta-prompt')) ?? '';
    assert.match(prompt, /litert-cookbook web skill/);
    assert.match(prompt, /model_web_a_example\.tflite/);
    await page.click('#cta-copy');

    const errors = (page as Page & { collectedErrors: string[] }).collectedErrors;
    assert.deepEqual(errors, [], `page errors: ${errors.join(' | ')}`);
    await page.close();
  });

  void test('model B: both backends run; fallback + unknown statuses render', async () => {
    const page = await openPlayground(context!, baseUrl);
    const requests = await checkFixture(page, 'model_web_b_example.tflite');
    assertNoModelBytesLeave(requests, baseUrl);

    assert.equal((await recordCells(page, 'wasm_xnnpack'))[2], 'yes', 'wasm runs');
    const webgpu = await recordCells(page, 'webgpu_mldrift');
    assert.equal(webgpu[2], 'yes', 'webgpu runs');
    assert.match(webgpu[4]!, /pass/, 'webgpu output match');

    // Known statuses: TANH=fallback and SQRT=partial from the snapshot; ABS
    // has no entry — unknown.
    const tanhCells = await page
      .locator('#xref-body tr', { hasText: 'TANH' })
      .locator('td')
      .allTextContents();
    assert.match(tanhCells[3]!, /fallback/);
    assert.match(tanhCells[3]!, /example rewrite hint for TANH/);
    const sqrtCells = await page
      .locator('#xref-body tr', { hasText: 'SQRT' })
      .locator('td')
      .allTextContents();
    assert.match(sqrtCells[3]!, /partial/);
    const absCells = await page
      .locator('#xref-body tr', { hasText: 'ABS' })
      .locator('td')
      .allTextContents();
    assert.match(absCells[2]!, /unknown/);
    assert.match(absCells[3]!, /unknown/);

    const errors = (page as Page & { collectedErrors: string[] }).collectedErrors;
    assert.deepEqual(errors, [], `page errors: ${errors.join(' | ')}`);
    await page.close();
  });

  void test('broken fixture: honest failure on every backend, nothing leaves', async () => {
    const page = await openPlayground(context!, baseUrl);
    const requests = await checkFixture(page, 'model_broken_example.tflite');
    assertNoModelBytesLeave(requests, baseUrl);

    // Operator inventory: parse failure is stated, not glossed over.
    const xref = (await page.textContent('#xref-body')) ?? '';
    assert.match(xref, /operator inventory unavailable/);
    assert.match(xref, /TFLite file identifier/);

    // Live records: loads=no with the parse_error failure class on both.
    for (const backend of ['wasm_xnnpack', 'webgpu_mldrift']) {
      const cells = await recordCells(page, backend);
      assert.equal(cells[1], 'no', `${backend} loads`);
      assert.equal(cells[2], 'no', `${backend} runs`);
      assert.match(cells[6]!, /parse_error/, `${backend} failure class`);
      assert.match(cells[6]!, /could not be parsed/, `${backend} plain-language explanation`);
    }
    const status = (await page.textContent('#pg-status')) ?? '';
    assert.match(status, /ran on 0 of 2 backend\(s\)/);

    const errors = (page as Page & { collectedErrors: string[] }).collectedErrors;
    assert.deepEqual(errors, [], `page errors: ${errors.join(' | ')}`);
    await page.close();
  });
});
