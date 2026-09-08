/** Unit tests for the probe driver's pure pieces (no browser launched). */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { litertCoreVersion, parseJob } from '../src/node/probe.ts';

const GOOD_JOB = {
  modelPath: '/tmp/fixture.tflite',
  accelerator: 'webgpu',
  inputs: [{ shape: [1, 4], dtype: 'float32', data: [0.5, 0.75, 1, 1.25] }],
  timeoutMs: 60000,
};

void test('parseJob accepts a well-formed job verbatim', () => {
  const job = parseJob(GOOD_JOB);
  assert.equal(job.accelerator, 'webgpu');
  assert.deepEqual(job.inputs[0]!.shape, [1, 4]);
  assert.equal(job.timeoutMs, 60000);
});

void test('parseJob rejects unknown accelerators', () => {
  assert.throws(() => parseJob({ ...GOOD_JOB, accelerator: 'webnn' }), /accelerator/);
});

void test('parseJob rejects non-float32-int32 input dtypes', () => {
  const inputs = [{ shape: [1], dtype: 'float64', data: [1] }];
  assert.throws(() => parseJob({ ...GOOD_JOB, inputs }), /inputs entries/);
});

void test('parseJob rejects empty inputs, bad shapes, and short timeouts', () => {
  assert.throws(() => parseJob({ ...GOOD_JOB, inputs: [] }), /non-empty/);
  const badShape = [{ shape: [0], dtype: 'float32', data: [1] }];
  assert.throws(() => parseJob({ ...GOOD_JOB, inputs: badShape }), /inputs entries/);
  assert.throws(() => parseJob({ ...GOOD_JOB, timeoutMs: 10 }), /timeoutMs/);
  assert.throws(() => parseJob({ ...GOOD_JOB, modelPath: '' }), /modelPath/);
});

void test('litertCoreVersion reports the installed @litertjs/core semver', () => {
  assert.match(litertCoreVersion(), /^\d+\.\d+\.\d+/);
});
