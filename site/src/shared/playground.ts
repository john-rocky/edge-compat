/**
 * Playground shared logic (Phase 9): types crossing the generator/page
 * boundary, the op × matrix cross-reference, the output comparator, and the
 * failure-class vocabulary — all pure and Node-testable.
 *
 * Verdict rule (spec 9.2, tested): every status the cross-reference emits is
 * either copied verbatim from an exported matrix snapshot entry or the literal
 * 'unknown'. Nothing here computes, infers, or defaults a verdict — that is
 * the Python linter's job, and reimplementing it is out of scope until a
 * golden-parity harness exists.
 */

/** Reference to one exported web-backend matrix snapshot (verbatim copy). */
export interface MatrixSnapshotRef {
  backend: string;
  /** For browser backends this is the @litertjs/core version (version-axis rule). */
  litertVersion: string;
  generatedAt: string;
  entryCount: number;
  /** URL of the verbatim snapshot JSON, relative to the playground page. */
  href: string;
  /**
   * The latest known @litertjs/core release when this snapshot's verified
   * version lags it (Phase 11, computed at build time); null when current or
   * unknown. Display only — the snapshot's verdicts are unchanged.
   */
  staleAgainst: { version: string; checkedAt: string } | null;
}

export interface PlaygroundConfig {
  /** Backends the playground runs live, in run order (reference first). */
  backends: string[];
  matrix: MatrixSnapshotRef[];
  warmupRuns: number;
  timedRuns: number;
  inputSeed: number;
  toleranceAbs: number;
  toleranceRel: number;
  /** Relative URL (from the playground page) of the @litertjs/core WASM assets. */
  wasmBaseUrl: string;
  litertjsVersion: string;
  /** One-line "reproduce this with an agent" prompt (Phase 12.2): the same
   * static verdicts locally via edge-lint, real commands only. */
  reproPrompt: string;
  repoUrl: string | null;
  cookbookUrl: string | null;
}

/** One matrix entry, exactly as it appears in the snapshot JSON. */
export interface MatrixEntry {
  op: string;
  dtypes?: string[];
  constraints?: Record<string, string | number | boolean>;
  status: string;
  conditions?: string;
  rewrite_hints?: { symptom: string; rewrite: string; expected_effect?: string }[];
  provenance: string;
  evidence?: Record<string, unknown>;
}

/** The subset of a matrix snapshot document the cross-reference reads. */
export interface MatrixSnapshotDoc {
  backend: string;
  litert_version: string;
  generated_at: string;
  entries: MatrixEntry[];
}

/** One op × one backend: the matching snapshot entries, verbatim. */
export interface CrossRefCell {
  backend: string;
  /**
   * Entries of the backend's snapshot whose `op` equals this row's op, in
   * snapshot order. Empty means the verdict is 'unknown' — either no snapshot
   * is exported for the backend or the snapshot has no entry for the op.
   */
  entries: MatrixEntry[];
  /** False when no snapshot is exported for this backend at all. */
  hasSnapshot: boolean;
}

export interface CrossRefRow {
  op: string;
  customCode: string | null;
  count: number;
  cells: CrossRefCell[];
}

/**
 * Cross-reference an operator inventory against exported matrix snapshots.
 * Op-name lookup only (the page knows op names, not per-node dtypes/shapes):
 * ALL entries for the op are surfaced verbatim so dtype/constraint context
 * stays visible; no entry-precedence logic runs client-side (DECISIONS #94).
 */
export function crossReference(
  inventory: readonly { op: string; customCode: string | null; count: number }[],
  backends: readonly string[],
  snapshots: ReadonlyMap<string, MatrixSnapshotDoc>,
): CrossRefRow[] {
  return inventory.map((item) => ({
    op: item.op,
    customCode: item.customCode,
    count: item.count,
    cells: backends.map((backend) => {
      const snapshot = snapshots.get(backend);
      return {
        backend,
        entries: snapshot === undefined ? [] : snapshot.entries.filter((e) => e.op === item.op),
        hasSnapshot: snapshot !== undefined,
      };
    }),
  }));
}

/**
 * Plain-language explanations for the sweep failure-class vocabulary
 * (sweep_result.schema.json backend_record.failure_class, open set), plus the
 * playground-only 'unsupported_input' class.
 */
export const FAILURE_EXPLANATIONS: Record<string, string> = {
  init_failed: 'The LiteRT.js runtime itself failed to load in this browser.',
  fetch_failed: 'The model bytes could not be read.',
  parse_error:
    'The file could not be parsed as a TFLite flatbuffer — is it really a .tflite export?',
  compile_error: 'The model parsed, but this backend failed to compile it.',
  wasm_memory_ceiling:
    'The model does not fit in browser WASM memory (about 2 GB addressable). ' +
    'This is a hard platform ceiling — it fails on any hardware, and quantizing ' +
    'or shrinking the model is the only way past it.',
  run_error: 'The model compiled, but an inference did not complete.',
  backend_unavailable:
    'This backend is not available in this browser (for WebGPU: unsupported or disabled).',
  unsupported_input:
    'Sample inputs could not be generated for this model (dynamic input ' +
    'dimensions, or an input dtype LiteRT.js tensors do not support). The ' +
    'playground runs random fixture inputs only.',
};

/**
 * Failure classification for load/compile errors. Same patterns as
 * web/sweep/src/node/sweep.ts classifyLoadError, so a playground failure class
 * matches what a sweep of the same model would record.
 */
export function classifyLoadError(message: string): string {
  const lower = message.toLowerCase();
  if (/(404|failed to fetch|networkerror|net::err)/.test(lower)) {
    return 'fetch_failed';
  }
  if (/(out of memory|memory access out of bounds|cannot allocate|allocation failed|oom)/.test(lower)) {
    return 'wasm_memory_ceiling';
  }
  if (/(flatbuffer|verif|invalid model|parse|malformed|not a valid|failed to load model)/.test(lower)) {
    return 'parse_error';
  }
  return 'compile_error';
}

export interface OutputComparison {
  match: boolean | null;
  maxAbsDiff: number | null;
  maxRelDiff: number | null;
  note: string | null;
}

const REL_DIFF_FLOOR = 1e-9;

/**
 * Output comparison against the wasm_xnnpack reference — the sweep's exact
 * semantics (web/sweep/src/node/results.ts compareOutputs): pass iff every
 * element satisfies |out - ref| <= max(tolAbs, tolRel * |ref|).
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
      note: `output tensor count mismatch: ${String(outputs.length)} vs reference ${String(reference.length)}`,
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
        note: `output[${String(t)}] element count mismatch: ${String(out.length)} vs reference ${String(ref.length)}`,
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
          note: `non-finite value in output[${String(t)}][${String(i)}] (out=${String(a)}, ref=${String(b)})`,
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

/** "Build this yourself" prompt for the visitor's own model. */
export function playgroundCtaPrompt(fileName: string): string {
  return (
    `Use the litert-cookbook web skill to build a LiteRT.js browser demo for my ` +
    `model ${fileName}: load the .tflite from a local path and run it with @litertjs/core.`
  );
}

/** The local full-analysis command the results page links to (spec 9.3). */
export function lintCommand(fileName: string, backend: string, snapshot: MatrixSnapshotRef | null): string {
  const matrixArg = snapshot === null ? 'data/matrix/<snapshot>.json' : `data/matrix/${backend}__${snapshot.litertVersion}.json`;
  return `edge-lint ${fileName} --backend ${backend} --matrix ${matrixArg} --md`;
}
