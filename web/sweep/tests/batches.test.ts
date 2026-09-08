/**
 * Batch driver: a fresh harness process per batch, a dying batch retried one
 * model at a time, the model that dies alone reported as unmeasured, and a
 * re-run that skips models already measured.
 */
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { runBatches } from '../src/node/batches.ts';

// Catalog order: example-web-a, example-broken, example-web-b.
const EXAMPLE_CATALOG = fileURLToPath(
  new URL('../../../data/examples/web_catalog_example.csv', import.meta.url),
);

function resultWriter(outDir: string): (id: string) => void {
  return (id) => {
    fs.writeFileSync(path.join(outDir, `${id}.json`), '{}');
  };
}

void describe('batch driver', () => {
  void test('a dying batch is retried one model at a time; the killer is unmeasured', () => {
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-batches-'));
    const write = resultWriter(outDir);
    const calls: string[][] = [];
    const run = (ids: readonly string[]): number | null => {
      calls.push([...ids]);
      const first = ids[0] ?? '';
      if (ids.length === 2) {
        write(first); // the first model lands, then the process dies
        return null;
      }
      if (first === 'example-broken') {
        return null; // dies alone too
      }
      write(first);
      return 0;
    };
    const lines: string[] = [];
    const outcome = runBatches({ catalogPath: EXAMPLE_CATALOG, outDir, batchSize: 2 }, run, (l) =>
      lines.push(l),
    );
    assert.deepEqual(calls, [
      ['example-web-a', 'example-broken'],
      ['example-broken'],
      ['example-web-b'],
    ]);
    assert.deepEqual(outcome.unmeasured, ['example-broken']);
    assert.deepEqual(outcome.measured, ['example-web-a', 'example-web-b']);
    assert.equal(outcome.invalidRuns, 0);
    assert.ok(lines.some((line) => line.includes('UNMEASURED example-broken')));
    fs.rmSync(outDir, { recursive: true, force: true });
  });

  void test('a re-run skips measured models; a validation exit is counted, not retried', () => {
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-batches-'));
    const write = resultWriter(outDir);
    write('example-web-a');
    const calls: string[][] = [];
    const run = (ids: readonly string[]): number | null => {
      calls.push([...ids]);
      for (const id of ids) {
        write(id);
      }
      return 1; // every file written, one of them schema-invalid
    };
    const outcome = runBatches({ catalogPath: EXAMPLE_CATALOG, outDir, batchSize: 8 }, run, () => {
      /* quiet */
    });
    assert.deepEqual(calls, [['example-broken', 'example-web-b']]);
    assert.deepEqual(outcome.unmeasured, []);
    assert.deepEqual(outcome.measured, ['example-web-a', 'example-broken', 'example-web-b']);
    assert.equal(outcome.invalidRuns, 1);
    fs.rmSync(outDir, { recursive: true, force: true });
  });
});
