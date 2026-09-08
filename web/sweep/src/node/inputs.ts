/**
 * Deterministic fixture inputs: same catalog + same seed => same input bytes,
 * on every platform (mulberry32 over IEEE-754 doubles, then fround for
 * float32). Input values feed the output-match diffs recorded in results, so
 * they must never depend on run order or wall clock.
 */
import * as fs from 'node:fs';

import type { FixtureInput } from '../shared/protocol.ts';
import { CatalogError, type TensorShapeSpec } from './catalog.ts';

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

export function buildFixtureInputs(specs: TensorShapeSpec[], seed: number): FixtureInput[] {
  return specs.map((spec, index) => {
    const rand = mulberry32(seed + index * 1000003);
    const count = spec.shape.reduce((a, b) => a * b, 1);
    const data: number[] = new Array<number>(count);
    for (let i = 0; i < count; i++) {
      data[i] = spec.dtype === 'int32' ? Math.floor(rand() * 16) : Math.fround(rand() * 2 - 1);
    }
    return { shape: spec.shape, dtype: spec.dtype, data };
  });
}

/** Load explicit inputs from a fixture JSON file: {"inputs": [{shape, dtype, data}]}. */
export function loadFixtureFile(filePath: string): FixtureInput[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (err) {
    throw new CatalogError(`cannot read input fixture file '${filePath}': ${String(err)}`);
  }
  if (typeof parsed !== 'object' || parsed === null || !('inputs' in parsed)) {
    throw new CatalogError(`input fixture file '${filePath}' must be {"inputs": [...]}`);
  }
  const inputs: unknown = parsed.inputs;
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new CatalogError(`input fixture file '${filePath}': "inputs" must be a non-empty array`);
  }
  return inputs.map((entry, index) => {
    const item = entry as Partial<FixtureInput>;
    if (
      !Array.isArray(item.shape) ||
      !item.shape.every((dim) => Number.isInteger(dim) && dim > 0) ||
      (item.dtype !== 'float32' && item.dtype !== 'int32') ||
      !Array.isArray(item.data) ||
      !item.data.every((value) => typeof value === 'number')
    ) {
      throw new CatalogError(
        `input fixture file '${filePath}': inputs[${index}] needs shape (positive ints), dtype (float32|int32), data (numbers)`,
      );
    }
    const count = item.shape.reduce((a, b) => a * b, 1);
    if (item.data.length !== count) {
      throw new CatalogError(
        `input fixture file '${filePath}': inputs[${index}] data length ${item.data.length} != shape volume ${count}`,
      );
    }
    return { shape: item.shape, dtype: item.dtype, data: item.data };
  });
}
