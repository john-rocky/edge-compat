/**
 * Playground unit tests (Phase 9): the op-inventory reader against the
 * committed fixtures, opcode-vocabulary parity with the Python parser, the
 * verdicts-only-from-the-matrix rule, snapshot loading, and the matrix-dir
 * build variant.
 */
import * as assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { loadWebMatrixSnapshots } from '../src/generator/matrix.ts';
import { BUILTIN_OPERATORS } from '../src/shared/opcodes.ts';
import {
  type MatrixSnapshotDoc,
  classifyLoadError,
  compareOutputs,
  crossReference,
  lintCommand,
} from '../src/shared/playground.ts';
import { TfliteParseError, readOpInventory } from '../src/shared/tflite.ts';

const SITE_ROOT = fileURLToPath(new URL('..', import.meta.url));
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const EXAMPLES = path.join(REPO_ROOT, 'data', 'examples');

function fixtureBytes(name: string): Uint8Array {
  return new Uint8Array(fs.readFileSync(path.join(EXAMPLES, name)));
}

void describe('opcode vocabulary parity with the Python parser', () => {
  void test('BUILTIN_OPERATORS matches src/litert_compat/parser/opcodes.py exactly', () => {
    const source = fs.readFileSync(
      path.join(REPO_ROOT, 'src', 'litert_compat', 'parser', 'opcodes.py'),
      'utf8',
    );
    const dictSource = /BUILTIN_OPERATORS: dict\[int, str\] = \{(.*?)\n\}/s.exec(source);
    assert.ok(dictSource !== null, 'BUILTIN_OPERATORS dict not found in opcodes.py');
    const python = new Map<number, string>();
    for (const match of dictSource[1]!.matchAll(/(\d+): "([A-Z0-9_]+)",/g)) {
      python.set(Number.parseInt(match[1]!, 10), match[2]!);
    }
    assert.ok(python.size > 200, `implausibly small Python table (${String(python.size)})`);
    assert.deepEqual(
      [...BUILTIN_OPERATORS.entries()].sort((a, b) => a[0] - b[0]),
      [...python.entries()].sort((a, b) => a[0] - b[0]),
      'TS opcode table drifted from the Python vocabulary — edit opcodes.py first, then mirror',
    );
  });
});

void describe('op inventory reader', () => {
  void test('reads the committed fixtures (matches the Python parser)', () => {
    assert.deepEqual(readOpInventory(fixtureBytes('model_web_a_example.tflite')), [
      { op: 'LOGISTIC', customCode: null, count: 1 },
      { op: 'NEG', customCode: null, count: 1 },
    ]);
    assert.deepEqual(readOpInventory(fixtureBytes('model_web_b_example.tflite')), [
      { op: 'ABS', customCode: null, count: 1 },
      { op: 'SQRT', customCode: null, count: 1 },
      { op: 'TANH', customCode: null, count: 1 },
    ]);
    // Repeated ops aggregate; custom ops carry their custom_code.
    assert.deepEqual(readOpInventory(fixtureBytes('model_clean_example.tflite')), [
      { op: 'AVERAGE_POOL_2D', customCode: null, count: 1 },
      { op: 'CONV_2D', customCode: null, count: 2 },
    ]);
    const mixed = readOpInventory(fixtureBytes('model_mixed_example.tflite'));
    assert.deepEqual(mixed.find((entry) => entry.op === 'CUSTOM'), {
      op: 'CUSTOM',
      customCode: 'ExampleCustomOp',
      count: 1,
    });
  });

  void test('rejects non-TFLite bytes with TfliteParseError', () => {
    assert.throws(
      () => readOpInventory(fixtureBytes('model_broken_example.tflite')),
      TfliteParseError,
    );
    assert.throws(() => readOpInventory(new Uint8Array([1, 2, 3])), TfliteParseError);
  });
});

void describe('cross-reference: verdicts come ONLY from the exported matrix', () => {
  const snapshot: MatrixSnapshotDoc = {
    backend: 'webgpu_mldrift',
    litert_version: '2.5.3',
    generated_at: '2026-08-10',
    entries: [
      { op: 'LOGISTIC', status: 'delegated', provenance: 'example' },
      { op: 'LOGISTIC', dtypes: ['int8'], status: 'fallback', provenance: 'example' },
      { op: 'SOFTMAX', status: 'crash', provenance: 'example' },
    ],
  };
  const inventory = [
    { op: 'LOGISTIC', customCode: null, count: 2 },
    { op: 'NEG', customCode: null, count: 1 },
  ];

  void test('matching entries pass through verbatim (object identity)', () => {
    const rows = crossReference(
      inventory,
      ['wasm_xnnpack', 'webgpu_mldrift'],
      new Map([['webgpu_mldrift', snapshot]]),
    );
    assert.equal(rows.length, 2);
    const logistic = rows[0]!;
    const webgpuCell = logistic.cells[1]!;
    // Both LOGISTIC entries surface, in snapshot order, as the SAME objects —
    // nothing is computed, filtered by precedence, or rewritten client-side.
    assert.equal(webgpuCell.entries.length, 2);
    assert.equal(webgpuCell.entries[0], snapshot.entries[0]);
    assert.equal(webgpuCell.entries[1], snapshot.entries[1]);
    // SOFTMAX is in the snapshot but not the model: never surfaced.
    assert.ok(rows.every((row) => row.op !== 'SOFTMAX'));
  });

  void test('no entry, or no snapshot, means unknown — never a guess', () => {
    const rows = crossReference(
      inventory,
      ['wasm_xnnpack', 'webgpu_mldrift'],
      new Map([['webgpu_mldrift', snapshot]]),
    );
    const neg = rows[1]!;
    // NEG has no entry in the webgpu snapshot: empty cell (renders 'unknown').
    assert.deepEqual(neg.cells[1], { backend: 'webgpu_mldrift', entries: [], hasSnapshot: true });
    // wasm_xnnpack has no snapshot at all: empty cell, flagged as such.
    assert.deepEqual(neg.cells[0], { backend: 'wasm_xnnpack', entries: [], hasSnapshot: false });
  });
});

void describe('output comparator (sweep semantics)', () => {
  void test('pass/fail/non-finite mirror web/sweep compareOutputs', () => {
    assert.equal(compareOutputs([[1, 2]], [[1, 2]], 1e-5, 1e-3).match, true);
    assert.equal(compareOutputs([[1, 2.1]], [[1, 2]], 1e-5, 1e-3).match, false);
    assert.equal(compareOutputs([[1, Number.NaN]], [[1, 2]], 1e-5, 1e-3).match, null);
    assert.equal(compareOutputs([[1]], [[1, 2]], 1e-5, 1e-3).match, false);
    const diff = compareOutputs([[2.01]], [[2]], 1e-5, 1e-3);
    assert.equal(diff.match, false);
    assert.ok(Math.abs(diff.maxAbsDiff! - 0.01) < 1e-12);
    assert.ok(Math.abs(diff.maxRelDiff! - 0.005) < 1e-12);
  });
});

void describe('failure classification (sweep patterns)', () => {
  void test('classes match web/sweep classifyLoadError', () => {
    assert.equal(classifyLoadError('Failed to fetch'), 'fetch_failed');
    assert.equal(classifyLoadError('Aborted(out of memory)'), 'wasm_memory_ceiling');
    assert.equal(classifyLoadError('The model is not a valid flatbuffer'), 'parse_error');
    assert.equal(classifyLoadError('something else went wrong'), 'compile_error');
  });
});

void describe('matrix snapshot loading', () => {
  void test('keeps web backends, skips native, errors on duplicates', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-matrix-'));
    try {
      fs.copyFileSync(
        path.join(EXAMPLES, 'matrix_webgpu_mldrift_playground_example.json'),
        path.join(dir, 'a.json'),
      );
      // Native snapshot: not exported to the page.
      fs.copyFileSync(path.join(EXAMPLES, 'matrix_example.json'), path.join(dir, 'native.json'));
      const snapshots = loadWebMatrixSnapshots(dir);
      assert.deepEqual(
        snapshots.map((s) => ({ backend: s.backend, entryCount: s.entryCount })),
        [{ backend: 'webgpu_mldrift', entryCount: 3 }],
      );
      // A second snapshot for the same backend: version-selection policy is
      // pending (Phase 4 deferred resolver) — refuse, never silently pick.
      fs.copyFileSync(
        path.join(EXAMPLES, 'matrix_webgpu_mldrift_example.json'),
        path.join(dir, 'b.json'),
      );
      assert.throws(() => loadWebMatrixSnapshots(dir), /two matrix snapshots/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  void test('missing directory means zero snapshots (the honest default)', () => {
    assert.deepEqual(loadWebMatrixSnapshots(path.join(os.tmpdir(), 'litert-no-such-dir')), []);
  });
});

void describe('lint command CTA', () => {
  void test('names the real snapshot when one is exported', () => {
    assert.equal(
      lintCommand('my_model.tflite', 'webgpu_mldrift', {
        backend: 'webgpu_mldrift',
        litertVersion: '2.5.3',
        generatedAt: '2026-08-10',
        entryCount: 3,
        href: '../matrix/webgpu_mldrift.json',
        staleAgainst: null,
      }),
      'edge-lint my_model.tflite --backend webgpu_mldrift --matrix data/matrix/webgpu_mldrift__2.5.3.json --md',
    );
    assert.equal(
      lintCommand('my_model.tflite', 'webgpu_mldrift', null),
      'edge-lint my_model.tflite --backend webgpu_mldrift --matrix data/matrix/<snapshot>.json --md',
    );
  });
});

void describe('build with --matrix-dir', () => {
  void test('exports web snapshots verbatim and embeds them in the page config', () => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'litert-pg-build-'));
    try {
      const matrixDir = path.join(base, 'matrix');
      const outDir = path.join(base, 'dist');
      fs.mkdirSync(matrixDir);
      fs.copyFileSync(
        path.join(EXAMPLES, 'matrix_webgpu_mldrift_playground_example.json'),
        path.join(matrixDir, 'webgpu_mldrift__2.5.3.json'),
      );
      fs.copyFileSync(
        path.join(EXAMPLES, 'matrix_wasm_xnnpack_example.json'),
        path.join(matrixDir, 'wasm_xnnpack__2.5.3.json'),
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
        ],
        { cwd: SITE_ROOT, encoding: 'utf8' },
      );
      assert.equal(result.status, 0, `build failed:\n${result.stdout}\n${result.stderr}`);
      assert.match(result.stdout, /playground with 2 matrix snapshot\(s\)/);

      // Snapshot copies are byte-verbatim.
      for (const [src, out] of [
        ['webgpu_mldrift__2.5.3.json', 'webgpu_mldrift.json'],
        ['wasm_xnnpack__2.5.3.json', 'wasm_xnnpack.json'],
      ] as const) {
        assert.ok(
          fs.readFileSync(path.join(matrixDir, src)).equals(
            fs.readFileSync(path.join(outDir, 'matrix', out)),
          ),
          `${out} is not a verbatim copy`,
        );
      }

      const html = fs.readFileSync(path.join(outDir, 'check', 'index.html'), 'utf8');
      const match = /<script type="application\/json" id="playground-config">(.*?)<\/script>/s.exec(
        html,
      );
      assert.ok(match !== null, 'playground-config block missing');
      const config = JSON.parse(match[1]!) as {
        backends: string[];
        matrix: { backend: string; href: string; entryCount: number }[];
      };
      assert.deepEqual(config.backends, ['wasm_xnnpack', 'webgpu_mldrift']);
      assert.deepEqual(
        config.matrix.map((m) => ({ backend: m.backend, href: m.href, entryCount: m.entryCount })),
        [
          { backend: 'wasm_xnnpack', href: '../matrix/wasm_xnnpack.json', entryCount: 0 },
          { backend: 'webgpu_mldrift', href: '../matrix/webgpu_mldrift.json', entryCount: 3 },
        ],
      );
      // The static snapshot table lists both, including the empty one.
      assert.match(html, /wasm_xnnpack/);
      assert.match(html, /webgpu_mldrift/);
    } finally {
      fs.rmSync(base, { recursive: true, force: true });
    }
  });

});
