/**
 * Model download cache (Phase 11): the DoD's caching demonstration — the
 * first run downloads, the second run hits the cache and makes no network
 * request. Also: atomic writes and URL-addressed identity.
 */
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as os from 'node:os';
import * as path from 'node:path';
import { after, before, describe, test } from 'node:test';

import { cachePathFor, ensureCached, trimCache } from '../src/node/cache.ts';

const BODY = Buffer.from('example-model-bytes-not-a-real-model');

let server: http.Server;
let baseUrl = '';
let requestCount = 0;
let cacheDir = '';

void describe('download cache', () => {
  void before(async () => {
    cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-cache-'));
    server = http.createServer((req, res) => {
      requestCount += 1;
      if (req.url === '/missing.tflite') {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' }).end(BODY);
    });
    await new Promise<void>((resolve) => {
      server.listen(0, '127.0.0.1', resolve);
    });
    const address = server.address();
    if (address === null || typeof address === 'string') {
      throw new Error('server failed to bind');
    }
    baseUrl = `http://127.0.0.1:${String(address.port)}`;
  });

  void after(async () => {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
    fs.rmSync(cacheDir, { recursive: true, force: true });
  });

  void test('second run hits the cache: one download, zero further requests', async () => {
    const url = `${baseUrl}/model.tflite`;
    const first = await ensureCached(url, cacheDir);
    assert.equal(first.hit, false);
    assert.equal(requestCount, 1);
    assert.ok(fs.readFileSync(first.path).equals(BODY));

    const second = await ensureCached(url, cacheDir);
    assert.equal(second.hit, true);
    assert.equal(second.path, first.path);
    assert.equal(requestCount, 1, 'cache hit must not touch the network');
    assert.ok(fs.readFileSync(second.path).equals(BODY));
  });

  void test('cache identity is the source URL', async () => {
    const urlA = `${baseUrl}/model.tflite`;
    const urlB = `${baseUrl}/other.tflite`;
    assert.notEqual(cachePathFor(cacheDir, urlA), cachePathFor(cacheDir, urlB));
    const result = await ensureCached(urlB, cacheDir);
    assert.equal(result.hit, false);
  });

  void test('a failed download caches nothing (no half-written hit)', async () => {
    const url = `${baseUrl}/missing.tflite`;
    await assert.rejects(ensureCached(url, cacheDir), /HTTP 404/);
    assert.equal(fs.existsSync(cachePathFor(cacheDir, url)), false);
    // Still a miss next time: the failure was not recorded as content.
    await assert.rejects(ensureCached(url, cacheDir), /HTTP 404/);
  });

  void test('a cache hit refreshes mtime, so LRU eviction sees the model as used', async () => {
    const url = `${baseUrl}/model.tflite`; // cached by the first test
    const cached = cachePathFor(cacheDir, url);
    const past = new Date(Date.now() - 3_600_000);
    fs.utimesSync(cached, past, past);
    await ensureCached(url, cacheDir);
    assert.ok(fs.statSync(cached).mtimeMs > past.getTime() + 1000);
  });

  void test('trimCache evicts least recently used first, never the kept file, to the budget', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-trim-'));
    const put = (name: string, bytes: number, ageSeconds: number): string => {
      const filePath = path.join(dir, `${name}.tflite`);
      fs.writeFileSync(filePath, Buffer.alloc(bytes));
      const when = new Date(Date.now() - ageSeconds * 1000);
      fs.utimesSync(filePath, when, when);
      return filePath;
    };
    const oldest = put('a', 100, 300);
    const middle = put('b', 100, 200);
    const newest = put('c', 100, 100);
    fs.writeFileSync(path.join(dir, 'x.tmp-1'), Buffer.alloc(50)); // not a model: ignored
    assert.deepEqual(trimCache(dir, 200), [oldest]);
    assert.ok(!fs.existsSync(oldest) && fs.existsSync(middle) && fs.existsSync(newest));
    assert.deepEqual(trimCache(dir, 0, middle), [newest]); // budget 0, `middle` in use
    assert.ok(fs.existsSync(middle));
    assert.deepEqual(trimCache(dir, 0), [middle]);
    assert.deepEqual(trimCache(path.join(dir, 'absent'), 0), []);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
