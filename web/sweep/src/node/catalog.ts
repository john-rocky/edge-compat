/**
 * Catalog manifest parsing (`data/web_catalog.csv` contract).
 *
 * Columns, exactly: model_id, source (url|path), input_spec, license, notes.
 * Local-path sources and input-fixture files resolve relative to the manifest
 * file. License is required per entry. Forgiving on whitespace, strict on
 * content — a malformed manifest is a usage error, not a sweep result.
 */
import * as path from 'node:path';

export const CATALOG_COLUMNS = ['model_id', 'source', 'input_spec', 'license', 'notes'] as const;

const MODEL_ID_PATTERN = /^[a-z0-9][a-z0-9._-]*$/;

export class CatalogError extends Error {}

export interface CatalogEntry {
  modelId: string;
  /** URL (http/https) or absolute local path, resolved against the manifest dir. */
  source: string;
  sourceIsUrl: boolean;
  /** Verbatim manifest cell (echoed into sweep results). */
  rawSource: string;
  inputSpec: string;
  license: string;
  notes: string;
}

/**
 * Minimal RFC 4180 parser: quoted cells, escaped quotes, CRLF/LF rows.
 * Lines whose first character is `#` are full-line comments (the catalog's
 * exclusion blocks, e.g. the litertlm section) and produce no row.
 */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;
  let sawAny = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '#' && !inQuotes && !sawAny && cell === '' && row.length === 0) {
      while (i + 1 < text.length && text[i + 1] !== '\n' && text[i + 1] !== '\r') {
        i++;
      }
      continue;
    }
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      sawAny = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
      sawAny = true;
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') {
        i++;
      }
      if (sawAny || cell.length > 0) {
        row.push(cell);
        rows.push(row);
      }
      row = [];
      cell = '';
      sawAny = false;
    } else {
      cell += ch;
      sawAny = true;
    }
  }
  if (inQuotes) {
    throw new CatalogError('unterminated quoted cell');
  }
  if (sawAny || cell.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function isUrl(source: string): boolean {
  return /^https?:\/\//.test(source);
}

export function parseCatalog(csvText: string, catalogDir: string): CatalogEntry[] {
  const rows = parseCsv(csvText);
  if (rows.length === 0) {
    throw new CatalogError('catalog is empty');
  }
  const header = (rows[0] ?? []).map((cell) => cell.trim());
  if (header.join(',') !== CATALOG_COLUMNS.join(',')) {
    throw new CatalogError(
      `catalog header must be exactly '${CATALOG_COLUMNS.join(',')}', got '${header.join(',')}'`,
    );
  }
  const entries: CatalogEntry[] = [];
  const seen = new Set<string>();
  for (let i = 1; i < rows.length; i++) {
    const cells = (rows[i] ?? []).map((cell) => cell.trim());
    if (cells.length === 1 && cells[0] === '') {
      continue;
    }
    if (cells.length !== CATALOG_COLUMNS.length) {
      throw new CatalogError(`row ${i + 1}: expected ${CATALOG_COLUMNS.length} cells, got ${cells.length}`);
    }
    const [modelId = '', rawSource = '', inputSpec = '', license = '', notes = ''] = cells;
    if (!MODEL_ID_PATTERN.test(modelId)) {
      throw new CatalogError(`row ${i + 1}: invalid model_id '${modelId}'`);
    }
    if (seen.has(modelId)) {
      throw new CatalogError(`row ${i + 1}: duplicate model_id '${modelId}'`);
    }
    seen.add(modelId);
    if (rawSource === '') {
      throw new CatalogError(`row ${i + 1} (${modelId}): source is required`);
    }
    if (license === '') {
      throw new CatalogError(`row ${i + 1} (${modelId}): license is required`);
    }
    if (inputSpec === '') {
      throw new CatalogError(`row ${i + 1} (${modelId}): input_spec is required`);
    }
    const sourceIsUrl = isUrl(rawSource);
    entries.push({
      modelId,
      source: sourceIsUrl ? rawSource : path.resolve(catalogDir, rawSource),
      sourceIsUrl,
      rawSource,
      inputSpec,
      license,
      notes,
    });
  }
  if (entries.length === 0) {
    throw new CatalogError('catalog has a header but no entries');
  }
  return entries;
}

export interface TensorShapeSpec {
  shape: number[];
  dtype: 'float32' | 'int32';
}

/** Parse a dimension spec like '1x64:float32' or '1x64:float32;1x8:int32'. */
export function parseShapeSpec(spec: string): TensorShapeSpec[] {
  const parts = spec.split(';').map((part) => part.trim());
  const result: TensorShapeSpec[] = [];
  for (const part of parts) {
    const match = /^(\d+(?:x\d+)*):(float32|int32)$/.exec(part);
    if (match === null) {
      throw new CatalogError(
        `invalid input_spec segment '${part}' (expected e.g. '1x64:float32'; dtypes: float32, int32)`,
      );
    }
    const dims = match[1]!.split('x').map((dim) => Number.parseInt(dim, 10));
    result.push({ shape: dims, dtype: match[2] as 'float32' | 'int32' });
  }
  return result;
}

/** True when the input_spec cell names an input-fixture JSON file. */
export function isFixtureFileSpec(spec: string): boolean {
  return spec.endsWith('.json');
}
