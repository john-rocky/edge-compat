/**
 * Sweep record assembly helpers: canonical JSON (sorted keys, 2-space indent,
 * trailing newline — the repo-wide byte-determinism convention), the p50
 * statistic, and the output-vs-reference comparator.
 */

export function canonicalStringify(value: unknown): string {
  return `${JSON.stringify(sortValue(value), null, 2)}\n`;
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortValue);
  }
  if (typeof value === 'object' && value !== null) {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      sorted[key] = sortValue((value as Record<string, unknown>)[key]);
    }
    return sorted;
  }
  return value;
}

/** Median: average of the two middle values for even counts. Rounded to 3 decimals. */
export function p50(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('p50 of empty array');
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
  return Math.round(median * 1000) / 1000;
}

export interface OutputComparison {
  /** Verdict for the record's output_match field (null = not comparable). */
  match: boolean | null;
  maxAbsDiff: number | null;
  maxRelDiff: number | null;
  /** Human-readable reason when match is null or structurally false. */
  note: string | null;
}

const REL_DIFF_FLOOR = 1e-9;

/**
 * Compare outputs against the wasm_xnnpack reference:
 * pass iff every element satisfies |out - ref| <= max(tolAbs, tolRel * |ref|).
 */
export function compareOutputs(
  outputs: readonly (readonly number[])[],
  reference: readonly (readonly number[])[],
  toleranceAbs: number,
  toleranceRel: number,
): OutputComparison {
  if (outputs.length !== reference.length) {
    return {
      match: false,
      maxAbsDiff: null,
      maxRelDiff: null,
      note: `output tensor count mismatch: ${outputs.length} vs reference ${reference.length}`,
    };
  }
  let maxAbs = 0;
  let maxRel = 0;
  let pass = true;
  for (let t = 0; t < outputs.length; t++) {
    const out = outputs[t]!;
    const ref = reference[t]!;
    if (out.length !== ref.length) {
      return {
        match: false,
        maxAbsDiff: null,
        maxRelDiff: null,
        note: `output[${t}] element count mismatch: ${out.length} vs reference ${ref.length}`,
      };
    }
    for (let i = 0; i < out.length; i++) {
      const a = out[i]!;
      const b = ref[i]!;
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        return {
          match: null,
          maxAbsDiff: null,
          maxRelDiff: null,
          note: `non-finite value in output[${t}][${i}] (out=${String(a)}, ref=${String(b)})`,
        };
      }
      const abs = Math.abs(a - b);
      const rel = abs / Math.max(Math.abs(b), REL_DIFF_FLOOR);
      maxAbs = Math.max(maxAbs, abs);
      maxRel = Math.max(maxRel, rel);
      if (abs > Math.max(toleranceAbs, toleranceRel * Math.abs(b))) {
        pass = false;
      }
    }
  }
  return { match: pass, maxAbsDiff: maxAbs, maxRelDiff: maxRel, note: null };
}
