/**
 * Web-backend matrix snapshot export for the playground (Phase 9).
 *
 * Scans a matrix directory (default: data/matrix/) for snapshot JSON files,
 * keeps the WEB backend snapshots (wasm_xnnpack / webgpu_mldrift / webnn —
 * the IDs registered in matrix.schema.json 1.1), and returns them for
 * verbatim copying into the site output. Native-backend snapshots are the
 * Python tools' concern and are not exported to the page.
 *
 * The playground's op cross-reference reads ONLY these exported files; with
 * data/matrix/ still empty, zero snapshots export and every op honestly
 * renders 'unknown'.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

import { SiteDataError } from './data.ts';

export const WEB_BACKENDS = ['wasm_xnnpack', 'webgpu_mldrift', 'webnn'] as const;

export interface WebMatrixSnapshot {
  backend: string;
  litertVersion: string;
  generatedAt: string;
  entryCount: number;
  /** Absolute path of the snapshot file; copied byte-verbatim. */
  abs: string;
}

export function loadWebMatrixSnapshots(matrixDir: string): WebMatrixSnapshot[] {
  if (!fs.existsSync(matrixDir)) {
    return [];
  }
  const snapshots: WebMatrixSnapshot[] = [];
  for (const name of fs.readdirSync(matrixDir).sort()) {
    if (!name.endsWith('.json')) {
      continue;
    }
    const abs = path.join(matrixDir, name);
    let doc: unknown;
    try {
      doc = JSON.parse(fs.readFileSync(abs, 'utf8'));
    } catch (err) {
      throw new SiteDataError(`${abs} is not valid JSON: ${String(err)}`);
    }
    if (typeof doc !== 'object' || doc === null || Array.isArray(doc)) {
      throw new SiteDataError(`${abs}: expected a matrix snapshot object`);
    }
    const snapshot = doc as Record<string, unknown>;
    const backend = snapshot['backend'];
    if (typeof backend !== 'string') {
      throw new SiteDataError(`${abs}: 'backend' must be a string`);
    }
    if (!(WEB_BACKENDS as readonly string[]).includes(backend)) {
      continue;
    }
    const litertVersion = snapshot['litert_version'];
    const generatedAt = snapshot['generated_at'];
    const entries = snapshot['entries'];
    if (typeof litertVersion !== 'string' || typeof generatedAt !== 'string') {
      throw new SiteDataError(`${abs}: 'litert_version' and 'generated_at' must be strings`);
    }
    if (!Array.isArray(entries)) {
      throw new SiteDataError(`${abs}: 'entries' must be an array`);
    }
    const duplicate = snapshots.find((s) => s.backend === backend);
    if (duplicate !== undefined) {
      // Which snapshot a backend's verdicts should come from when several
      // versions exist is a version-selection policy the owner has not set
      // yet (Phase 4 deferred backend→snapshot resolver). Refuse rather than
      // silently pick one.
      throw new SiteDataError(
        `two matrix snapshots for backend '${backend}' (${path.basename(duplicate.abs)} and ` +
          `${name}) — version-selection policy is pending; keep one snapshot per web backend ` +
          `in the matrix directory`,
      );
    }
    snapshots.push({
      backend,
      litertVersion,
      generatedAt,
      entryCount: entries.length,
      abs,
    });
  }
  return snapshots;
}
