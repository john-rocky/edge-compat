/**
 * Model download cache (Phase 11): URL-sourced catalog models are downloaded
 * once into a cache directory and served to the harness page from the local
 * server on later runs. This is what makes the weekly full-catalog re-sweep
 * affordable — in CI the directory is persisted with actions/cache, keyed by
 * the catalog file's hash, so a re-run of an unchanged catalog downloads
 * nothing.
 *
 * Cache entries are content-addressed by the SOURCE URL's sha256 (the URL is
 * the identity the catalog declares; a changed URL is a different artifact).
 * Writes are atomic (temp file + rename) so an interrupted download never
 * leaves a half-written model to be replayed as a cache hit.
 *
 * The cache can be bounded (`trimCache`): the real catalog is 33 GB, which
 * fits neither a hosted CI runner's disk nor GitHub's 10 GB cache — the
 * 2026-08-27 run filled the disk pre-downloading it — so CI stages one model
 * at a time with a budget of 0 and only a workstation keeps everything.
 */
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import type { ReadableStream as WebReadableStream } from 'node:stream/web';

export interface CacheResult {
  /** Absolute path of the cached model file. */
  path: string;
  /** True when the file was already present (no network request made). */
  hit: boolean;
}

export function cachePathFor(cacheDir: string, url: string): string {
  const digest = createHash('sha256').update(url).digest('hex');
  return path.join(cacheDir, `${digest}.tflite`);
}

export async function ensureCached(
  url: string,
  cacheDir: string,
  fetchImpl: typeof fetch = fetch,
): Promise<CacheResult> {
  const target = cachePathFor(cacheDir, url);
  if (fs.existsSync(target)) {
    // A hit counts as a use: `trimCache` evicts by mtime, oldest first.
    const now = new Date();
    fs.utimesSync(target, now, now);
    return { path: target, hit: true };
  }
  fs.mkdirSync(cacheDir, { recursive: true });
  const response = await fetchImpl(url);
  if (!response.ok) {
    throw new Error(`download failed: ${url} -> HTTP ${String(response.status)}`);
  }
  if (response.body === null) {
    throw new Error(`download failed: ${url} -> empty response body`);
  }
  // Stream to disk: Buffer.from(arrayBuffer) caps at 2^31-1 bytes, which a
  // >2 GB artifact (e.g. a 1B fp16 embed model) exceeds.
  const temp = `${target}.tmp-${String(process.pid)}`;
  await pipeline(
    Readable.fromWeb(response.body as WebReadableStream),
    fs.createWriteStream(temp),
  );
  fs.renameSync(temp, target);
  return { path: target, hit: false };
}

/**
 * Evict cached models, least recently used (by mtime) first, until the cache
 * holds at most `maxBytes` — never evicting `keep` (the model about to be
 * served). Returns the evicted paths. `maxBytes` 0 keeps nothing but `keep`.
 */
export function trimCache(cacheDir: string, maxBytes: number, keep?: string): string[] {
  if (!fs.existsSync(cacheDir)) {
    return [];
  }
  const files = fs
    .readdirSync(cacheDir)
    .filter((name) => name.endsWith('.tflite'))
    .map((name) => {
      const filePath = path.join(cacheDir, name);
      const stat = fs.statSync(filePath);
      return { path: filePath, size: stat.size, mtimeMs: stat.mtimeMs };
    })
    .sort((a, b) => a.mtimeMs - b.mtimeMs);
  let total = files.reduce((sum, file) => sum + file.size, 0);
  const evicted: string[] = [];
  for (const file of files) {
    if (total <= maxBytes) {
      break;
    }
    if (file.path === keep) {
      continue;
    }
    fs.unlinkSync(file.path);
    total -= file.size;
    evicted.push(file.path);
  }
  return evicted;
}
