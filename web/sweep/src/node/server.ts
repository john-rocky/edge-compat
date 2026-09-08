/**
 * Local static server for the sweep: harness page + bundle, the
 * @litertjs/core WASM assets, and local-path model files from the catalog.
 *
 * COOP/COEP headers are set so the page is cross-origin isolated — this lets
 * LiteRT.js pick its most capable WASM variant (threads); URL-sourced models
 * must therefore be served with CORS headers (Hugging Face resolve URLs are).
 */
import * as fs from 'node:fs';
import * as http from 'node:http';
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

const SWEEP_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const WASM_DIR = path.join(path.dirname(require.resolve('@litertjs/core/package.json')), 'wasm');

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.tflite': 'application/octet-stream',
};

export interface SweepServer {
  baseUrl: string;
  close(): Promise<void>;
}

/** modelRoutes: model_id -> absolute local file path, served at /models/<model_id>. */
export async function startServer(modelRoutes: ReadonlyMap<string, string>): Promise<SweepServer> {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    const filePath = resolvePath(url.pathname, modelRoutes);
    if (filePath === null || !fs.existsSync(filePath)) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    fs.createReadStream(filePath).pipe(res);
  });

  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (address === null || typeof address === 'string') {
    throw new Error('server failed to bind');
  }
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}

function resolvePath(pathname: string, modelRoutes: ReadonlyMap<string, string>): string | null {
  if (pathname === '/') {
    return path.join(SWEEP_ROOT, 'src', 'page', 'index.html');
  }
  if (pathname === '/harness.js') {
    return path.join(SWEEP_ROOT, 'dist', 'page', 'harness.js');
  }
  if (pathname.startsWith('/wasm/')) {
    const name = path.basename(pathname);
    return path.join(WASM_DIR, name);
  }
  if (pathname.startsWith('/models/')) {
    const modelId = decodeURIComponent(pathname.slice('/models/'.length));
    return modelRoutes.get(modelId) ?? null;
  }
  return null;
}
