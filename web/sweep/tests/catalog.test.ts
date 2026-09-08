import * as assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  CatalogError,
  isFixtureFileSpec,
  parseCatalog,
  parseCsv,
  parseShapeSpec,
} from '../src/node/catalog.ts';

const HEADER = 'model_id,source,input_spec,license,notes';

void test('parseCsv handles quotes, escaped quotes, and CRLF', () => {
  const rows = parseCsv('a,"b,c",d\r\ne,"f""g",\n');
  assert.deepEqual(rows, [
    ['a', 'b,c', 'd'],
    ['e', 'f"g', ''],
  ]);
});

void test('parseCsv rejects unterminated quotes', () => {
  assert.throws(() => parseCsv('a,"b'), CatalogError);
});

void test('parseCsv skips full-line comments, including ones with commas', () => {
  const rows = parseCsv('# heading, with a comma\na,b,c\n# tail note\n');
  assert.deepEqual(rows, [['a', 'b', 'c']]);
});

void test('parseCsv keeps # that is not at line start literal', () => {
  const rows = parseCsv('a,#b,"c#d"\n');
  assert.deepEqual(rows, [['a', '#b', 'c#d']]);
});

void test('parseCatalog accepts the catalog exclusion-block convention', () => {
  const entries = parseCatalog(
    `${HEADER}\n# litertlm (LLM lane — cards/matrix only, not browser-sweepable)\nm1,model.tflite,1x64:float32,Apache-2.0,\n# Excluded, dynamic input dim: nima\n`,
    '/base',
  );
  assert.equal(entries.length, 1);
  assert.equal(entries[0]!.modelId, 'm1');
});

void test('parseCatalog resolves local paths relative to the catalog dir', () => {
  const entries = parseCatalog(`${HEADER}\nm1,model.tflite,1x64:float32,Apache-2.0,note\n`, '/base');
  assert.equal(entries.length, 1);
  assert.equal(entries[0]!.source, '/base/model.tflite');
  assert.equal(entries[0]!.rawSource, 'model.tflite');
  assert.equal(entries[0]!.sourceIsUrl, false);
});

void test('parseCatalog keeps URL sources verbatim', () => {
  const entries = parseCatalog(
    `${HEADER}\nm1,https://example.com/m.tflite,1x64:float32,Apache-2.0,\n`,
    '/base',
  );
  assert.equal(entries[0]!.source, 'https://example.com/m.tflite');
  assert.equal(entries[0]!.sourceIsUrl, true);
});

void test('parseCatalog enforces header, license, duplicate ids, id pattern', () => {
  assert.throws(() => parseCatalog('bad,header\nx,y\n', '/b'), CatalogError);
  assert.throws(
    () => parseCatalog(`${HEADER}\nm1,model.tflite,1x64:float32,,note\n`, '/b'),
    /license is required/,
  );
  assert.throws(
    () =>
      parseCatalog(
        `${HEADER}\nm1,a.tflite,1x64:float32,MIT,\nm1,b.tflite,1x64:float32,MIT,\n`,
        '/b',
      ),
    /duplicate model_id/,
  );
  assert.throws(
    () => parseCatalog(`${HEADER}\nBadId,a.tflite,1x64:float32,MIT,\n`, '/b'),
    /invalid model_id/,
  );
});

void test('parseShapeSpec parses single and multiple inputs', () => {
  assert.deepEqual(parseShapeSpec('1x64:float32'), [{ shape: [1, 64], dtype: 'float32' }]);
  assert.deepEqual(parseShapeSpec('1x8x8x3:float32; 4:int32'), [
    { shape: [1, 8, 8, 3], dtype: 'float32' },
    { shape: [4], dtype: 'int32' },
  ]);
});

void test('parseShapeSpec rejects malformed specs', () => {
  for (const bad of ['', '1x64', '1x64:float64', 'x64:float32', '1x-4:float32']) {
    assert.throws(() => parseShapeSpec(bad), CatalogError, bad);
  }
});

void test('isFixtureFileSpec distinguishes fixture files from shape specs', () => {
  assert.equal(isFixtureFileSpec('inputs/m1.json'), true);
  assert.equal(isFixtureFileSpec('1x64:float32'), false);
});
