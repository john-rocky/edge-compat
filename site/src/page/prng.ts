/**
 * Deterministic in-page sample inputs.
 *
 * Mirrors web/sweep/src/node/inputs.ts exactly (mulberry32 over IEEE-754
 * doubles, fround for float32, per-input seed offset), so a demo page run
 * feeds the model the same sample input the sweep measured it with.
 */
import type { TensorSpec } from '../shared/config.ts';

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SampleInput {
  shape: number[];
  data: Float32Array | Int32Array;
}

export function buildSampleInputs(specs: TensorSpec[], seed: number): SampleInput[] {
  return specs.map((spec, index) => {
    const rand = mulberry32(seed + index * 1000003);
    const count = spec.shape.reduce((a, b) => a * b, 1);
    if (spec.dtype === 'int32') {
      const data = new Int32Array(count);
      for (let i = 0; i < count; i++) {
        data[i] = Math.floor(rand() * 16);
      }
      return { shape: spec.shape, data };
    }
    const data = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      data[i] = Math.fround(rand() * 2 - 1);
    }
    return { shape: spec.shape, data };
  });
}

/** Median with the sweep's rounding (web/sweep/src/node/results.ts p50). */
export function p50(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('p50 of empty array');
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
  return Math.round(median * 1000) / 1000;
}
