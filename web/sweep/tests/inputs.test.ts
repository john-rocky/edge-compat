import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { test } from 'node:test';

import { CatalogError } from '../src/node/catalog.ts';
import { buildFixtureInputs, loadFixtureFile, mulberry32 } from '../src/node/inputs.ts';

void test('mulberry32 is deterministic for a given seed', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const seqA = [a(), a(), a()];
  const seqB = [b(), b(), b()];
  assert.deepEqual(seqA, seqB);
  for (const value of seqA) {
    assert.ok(value >= 0 && value < 1);
  }
});

void test('buildFixtureInputs: same specs + seed => identical data; ranges respected', () => {
  const specs = [
    { shape: [1, 8], dtype: 'float32' as const },
    { shape: [4], dtype: 'int32' as const },
  ];
  const first = buildFixtureInputs(specs, 42);
  const second = buildFixtureInputs(specs, 42);
  assert.deepEqual(first, second);
  assert.equal(first[0]!.data.length, 8);
  assert.equal(first[1]!.data.length, 4);
  for (const value of first[0]!.data) {
    assert.ok(value >= -1 && value < 1);
    assert.equal(value, Math.fround(value), 'float32-representable');
  }
  for (const value of first[1]!.data) {
    assert.ok(Number.isInteger(value) && value >= 0 && value < 16);
  }
  const reseeded = buildFixtureInputs(specs, 43);
  assert.notDeepEqual(first[0]!.data, reseeded[0]!.data);
});

void test('loadFixtureFile validates structure and volume', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sweep-inputs-'));
  const good = path.join(dir, 'good.json');
  fs.writeFileSync(
    good,
    JSON.stringify({ inputs: [{ shape: [1, 2], dtype: 'float32', data: [0.5, -0.5] }] }),
  );
  assert.deepEqual(loadFixtureFile(good), [{ shape: [1, 2], dtype: 'float32', data: [0.5, -0.5] }]);

  const badVolume = path.join(dir, 'bad_volume.json');
  fs.writeFileSync(
    badVolume,
    JSON.stringify({ inputs: [{ shape: [1, 3], dtype: 'float32', data: [1, 2] }] }),
  );
  assert.throws(() => loadFixtureFile(badVolume), /data length 2 != shape volume 3/);

  const badDtype = path.join(dir, 'bad_dtype.json');
  fs.writeFileSync(
    badDtype,
    JSON.stringify({ inputs: [{ shape: [1], dtype: 'float64', data: [1] }] }),
  );
  assert.throws(() => loadFixtureFile(badDtype), CatalogError);

  assert.throws(() => loadFixtureFile(path.join(dir, 'missing.json')), CatalogError);
});
