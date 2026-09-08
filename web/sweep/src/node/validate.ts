/**
 * Every sweep result file is validated against the committed
 * schemas/sweep_result.schema.json before the sweep exits — the schema, not
 * this harness, is the public contract (Cross-Phase Contracts rule).
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Ajv2020, type ValidateFunction } from 'ajv/dist/2020.js';

const REPO_ROOT = fileURLToPath(new URL('../../../..', import.meta.url));
export const SWEEP_SCHEMA_PATH = path.join(REPO_ROOT, 'schemas', 'sweep_result.schema.json');

let cachedValidator: ValidateFunction | null = null;

function validator(): ValidateFunction {
  if (cachedValidator !== null) {
    return cachedValidator;
  }
  const schema = JSON.parse(fs.readFileSync(SWEEP_SCHEMA_PATH, 'utf8')) as object;
  const ajv = new Ajv2020({ allErrors: true });
  const compiled = ajv.compile(schema);
  cachedValidator = compiled;
  return compiled;
}

/** Returns human-readable defects; empty array = valid. */
export function validateSweepResult(result: unknown): string[] {
  const validate = validator();
  if (validate(result)) {
    return [];
  }
  return (validate.errors ?? []).map(
    (err) => `${err.instancePath === '' ? '<root>' : err.instancePath} ${err.message ?? 'invalid'}`,
  );
}
