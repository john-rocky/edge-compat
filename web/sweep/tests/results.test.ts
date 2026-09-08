import * as assert from 'node:assert/strict';
import { test } from 'node:test';

import { canonicalStringify, compareOutputs, p50 } from '../src/node/results.ts';
import { validateSweepResult } from '../src/node/validate.ts';

void test('canonicalStringify sorts keys recursively, 2-space indent, trailing newline', () => {
  const text = canonicalStringify({ b: 1, a: { d: [{ z: 1, y: 2 }], c: 3 } });
  assert.equal(text, `{\n  "a": {\n    "c": 3,\n    "d": [\n      {\n        "y": 2,\n        "z": 1\n      }\n    ]\n  },\n  "b": 1\n}\n`);
  assert.equal(canonicalStringify({ b: 1, a: 2 }), canonicalStringify({ a: 2, b: 1 }));
});

void test('p50: odd takes middle, even averages the two middles', () => {
  assert.equal(p50([3, 1, 2]), 2);
  assert.equal(p50([4, 1, 3, 2]), 2.5);
  assert.equal(p50([1.23456]), 1.235);
  assert.throws(() => p50([]));
});

void test('compareOutputs passes within tolerance and fails outside it', () => {
  const within = compareOutputs([[1.0, 2.0]], [[1.0005, 2.0]], 1e-5, 1e-3);
  assert.equal(within.match, true);
  assert.ok(within.maxAbsDiff! > 0);
  assert.ok(within.maxRelDiff! > 0);

  const outside = compareOutputs([[1.1]], [[1.0]], 1e-5, 1e-3);
  assert.equal(outside.match, false);
  assert.ok(Math.abs(outside.maxAbsDiff! - 0.1) < 1e-12);
});

void test('compareOutputs: structural mismatch is false with null diffs; non-finite is null', () => {
  const countMismatch = compareOutputs([[1]], [[1], [2]], 1e-5, 1e-3);
  assert.equal(countMismatch.match, false);
  assert.equal(countMismatch.maxAbsDiff, null);
  assert.match(countMismatch.note ?? '', /tensor count mismatch/);

  const lengthMismatch = compareOutputs([[1, 2]], [[1]], 1e-5, 1e-3);
  assert.equal(lengthMismatch.match, false);
  assert.match(lengthMismatch.note ?? '', /element count mismatch/);

  const nonFinite = compareOutputs([[Number.NaN]], [[1]], 1e-5, 1e-3);
  assert.equal(nonFinite.match, null);
  assert.match(nonFinite.note ?? '', /non-finite/);
});

const VALID_RECORD = {
  backend: 'wasm_xnnpack',
  loads: true,
  runs: true,
  failure_class: null,
  error: null,
  full_delegation: null,
  output_match: null,
  max_abs_diff: null,
  max_rel_diff: null,
  latency_p50_ms: 1.234,
  delegation_evidence: [],
  env: {
    browser: 'chromium',
    browser_version: '139.0.0.0',
    headless: true,
    os: 'macOS',
    os_version: '27.0.0',
    jspi: true,
    webgpu_adapter: null,
    litertjs_core_version: '2.5.3',
    machine_label: 'test',
  },
  date: '2026-08-10',
  provenance: 'example',
};

const VALID_RESULT = {
  schema_version: '1.0',
  model_id: 'example-web-a',
  source: 'model_web_a_example.tflite',
  license: 'Apache-2.0',
  input_spec: '1x64:float32',
  notes: '',
  config: { warmup_runs: 3, timed_runs: 10, tolerance_abs: 1e-5, tolerance_rel: 1e-3, input_seed: 42 },
  results: [VALID_RECORD],
};

void test('validateSweepResult accepts a well-formed result file', () => {
  assert.deepEqual(validateSweepResult(VALID_RESULT), []);
});

void test('validateSweepResult rejects records without env (honesty rule)', () => {
  const { env: _env, ...withoutEnv } = VALID_RECORD;
  const invalid = { ...VALID_RESULT, results: [withoutEnv] };
  const errors = validateSweepResult(invalid);
  assert.ok(errors.length > 0);
  assert.match(errors.join('\n'), /env/);
});

void test('validateSweepResult rejects unknown backends and bad provenance', () => {
  const badBackend = { ...VALID_RESULT, results: [{ ...VALID_RECORD, backend: 'gpu_mldrift' }] };
  assert.ok(validateSweepResult(badBackend).length > 0);
  const badProvenance = { ...VALID_RESULT, results: [{ ...VALID_RECORD, provenance: 'inferred' }] };
  assert.ok(validateSweepResult(badProvenance).length > 0);
});
