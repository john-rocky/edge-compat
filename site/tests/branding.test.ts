/**
 * Branding parity (Phase 12.1): the TS constants in src/shared/branding.ts
 * mirror src/litert_compat/branding.py exactly, parsed from the Python
 * source — the DECISIONS #93 pattern. A rename or disclosure-wording change
 * edits the Python side first, then this test forces the mirror.
 */
import * as assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { DISCLOSURE_LINE, NAMING_NOTICE, PROJECT_NAME } from '../src/shared/branding.ts';

const REPO_ROOT = path.resolve(fileURLToPath(new URL('..', import.meta.url)), '..');

/** Extract a module-level string constant (plain or parenthesized implicit
 * concatenation) from Python source. */
function pythonStringConstant(source: string, name: string): string {
  const single = new RegExp(`^${name} = "([^"]*)"$`, 'm').exec(source);
  if (single !== null) {
    return single[1]!;
  }
  const multi = new RegExp(`^${name} = \\(\\n([\\s\\S]*?)^\\)$`, 'm').exec(source);
  assert.ok(multi !== null, `${name} not found in branding.py`);
  let value = '';
  for (const fragment of multi[1]!.matchAll(/"([^"]*)"/g)) {
    value += fragment[1]!;
  }
  assert.ok(value.length > 0, `${name} parsed empty from branding.py`);
  return value;
}

void describe('branding parity with src/litert_compat/branding.py (Phase 12.1)', () => {
  const source = fs.readFileSync(
    path.join(REPO_ROOT, 'src', 'litert_compat', 'branding.py'),
    'utf8',
  );

  void test('PROJECT_NAME, DISCLOSURE_LINE, NAMING_NOTICE match exactly', () => {
    assert.equal(PROJECT_NAME, pythonStringConstant(source, 'PROJECT_NAME'));
    assert.equal(DISCLOSURE_LINE, pythonStringConstant(source, 'DISCLOSURE_LINE'));
    assert.equal(NAMING_NOTICE, pythonStringConstant(source, 'NAMING_NOTICE'));
  });
});
