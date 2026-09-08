/**
 * Headless smoke test for one example demo page (Phase 7 DoD).
 *
 * Serves the built site over plain HTTP with NO COOP/COEP headers —
 * deliberately mirroring GitHub Pages, which cannot set them — and
 * intercepts the example model's https://example.invalid URL, fulfilling it
 * with the local synthetic fixture bytes. That exercises the real code path
 * (fetch model from an external URL at runtime) without hosting weights.
 */
import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as os from 'node:os';
import * as path from 'node:path';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { type Browser, type BrowserContext, chromium } from 'playwright';

const SITE_ROOT = fileURLToPath(new URL('..', import.meta.url));
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const FIXTURE = path.join(REPO_ROOT, 'data', 'examples', 'model_web_a_example.tflite');

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

function buildSite(outDir: string): void {
  const result = spawnSync(
    process.execPath,
    ['--experimental-strip-types', path.join('src', 'generator', 'cli.ts'), '--out', outDir],
    { cwd: SITE_ROOT, encoding: 'utf8' },
  );
  assert.equal(result.status, 0, `build failed:\n${result.stdout}\n${result.stderr}`);
}

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
    // NO COOP/COEP: GitHub Pages cannot set them either.
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

void describe('demo page smoke (headless, Pages-like: no COOP/COEP)', () => {
  let outDir = '';
  let baseUrl = '';
  let closeServer: (() => Promise<void>) | null = null;
  let browser: Browser | null = null;
  let context: BrowserContext | null = null;

  void before(async () => {
    outDir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'litert-site-smoke-')), 'dist');
    buildSite(outDir);
    const server = await startStaticServer(outDir);
    baseUrl = server.baseUrl;
    closeServer = server.close;
    browser = await chromium.launch({ headless: true });
    context = await browser.newContext();
    // The demo page fetches the model from its (fake) official source URL;
    // fulfill with the local synthetic fixture bytes, CORS-enabled.
    await context.route('https://example.invalid/**', (route) => {
      void route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Access-Control-Allow-Origin': '*',
        },
        body: fs.readFileSync(FIXTURE),
      });
    });
  });

  void after(async () => {
    await context?.close();
    await browser?.close();
    await closeServer?.();
    fs.rmSync(path.dirname(outDir), { recursive: true, force: true });
  });

  void test('example demo page loads the model, runs, and reports live latency', async () => {
    const page = await context!.newPage();
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    await page.goto(`${baseUrl}/models/example-web-a/`, { waitUntil: 'load' });
    await page.waitForSelector('body[data-demo-state="done"]', { timeout: 120000 });

    const backend = (await page.textContent('#live-backend'))?.trim() ?? '';
    const latency = (await page.textContent('#live-latency'))?.trim() ?? '';
    assert.match(backend, /wasm|webgpu/, `backend actually in use should be shown, got '${backend}'`);
    assert.match(latency, /^\d+(\.\d+)? ms$/, `live latency should be shown, got '${latency}'`);

    // Env-labeled sweep numbers are on the page next to the live run.
    const body = (await page.textContent('body')) ?? '';
    assert.match(body, /wasm_xnnpack/);
    assert.match(body, /example-dev-mac-arm64/);

    // CTA: visible one-line prompt + copy button.
    assert.equal(await page.isVisible('#cta-copy'), true);
    const prompt = (await page.textContent('#cta-prompt')) ?? '';
    assert.match(prompt, /litert-cookbook web skill/);
    assert.match(prompt, /example-web-a/);
    await page.click('#cta-copy');
    assert.equal(
      await page.getAttribute('body', 'data-demo-state'),
      'done',
      'CTA click must not disturb the finished demo state',
    );
    assert.deepEqual(errors, [], `page errors: ${errors.join(' | ')}`);
    await page.close();
  });

  void test('index page renders and sorts', async () => {
    const page = await context!.newPage();
    await page.goto(`${baseUrl}/`, { waitUntil: 'load' });
    assert.match((await page.title()) ?? '', /LiteRT\.js Demo Zoo/);
    assert.equal(await page.isVisible('table.sortable'), true);
    // Click the first backend column header; sorting must annotate aria-sort.
    const header = page.locator('table.sortable thead th').nth(3);
    await header.click();
    assert.equal(await header.getAttribute('aria-sort'), 'ascending');
    // Demo links exist for passing models only.
    assert.equal(await page.isVisible('a[href="models/example-web-a/"]'), true);
    await page.close();
  });
});
