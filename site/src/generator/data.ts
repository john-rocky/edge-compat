/**
 * Site data loading: cards/index.json + per-model card.json (+ the linked
 * sweep_source file for sample-input specs and sweep config).
 *
 * The generator consumes committed, already-validated artifacts (the Python
 * test suite byte-pins them), so validation here is structural with clear
 * errors — a malformed input aborts the build, it never renders a guess.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

import type { TensorSpec } from '../shared/config.ts';

export class SiteDataError extends Error {}

function asObject(value: unknown, ctx: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new SiteDataError(`${ctx}: expected an object`);
  }
  return value as Record<string, unknown>;
}

function getString(obj: Record<string, unknown>, key: string, ctx: string): string {
  const value = obj[key];
  if (typeof value !== 'string') {
    throw new SiteDataError(`${ctx}: '${key}' must be a string`);
  }
  return value;
}

function getNumber(obj: Record<string, unknown>, key: string, ctx: string): number {
  const value = obj[key];
  if (typeof value !== 'number') {
    throw new SiteDataError(`${ctx}: '${key}' must be a number`);
  }
  return value;
}

function getBoolean(obj: Record<string, unknown>, key: string, ctx: string): boolean {
  const value = obj[key];
  if (typeof value !== 'boolean') {
    throw new SiteDataError(`${ctx}: '${key}' must be a boolean`);
  }
  return value;
}

function getNullableNumber(
  obj: Record<string, unknown>,
  key: string,
  ctx: string,
): number | null {
  const value = obj[key];
  if (value !== null && typeof value !== 'number') {
    throw new SiteDataError(`${ctx}: '${key}' must be a number or null`);
  }
  return value;
}

function getNullableBoolean(
  obj: Record<string, unknown>,
  key: string,
  ctx: string,
): boolean | null {
  const value = obj[key];
  if (value !== null && typeof value !== 'boolean') {
    throw new SiteDataError(`${ctx}: '${key}' must be a boolean or null`);
  }
  return value;
}

function getArray(obj: Record<string, unknown>, key: string, ctx: string): unknown[] {
  const value = obj[key];
  if (!Array.isArray(value)) {
    throw new SiteDataError(`${ctx}: '${key}' must be an array`);
  }
  return value;
}

function readJson(filePath: string): unknown {
  let text: string;
  try {
    text = fs.readFileSync(filePath, 'utf8');
  } catch (err) {
    throw new SiteDataError(`cannot read ${filePath}: ${String(err)}`);
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new SiteDataError(`${filePath} is not valid JSON: ${String(err)}`);
  }
}

/** One record of a card's `browser.backends` array, as displayed by the site. */
export interface CardBrowserRecord {
  backend: string;
  date: string;
  fullDelegation: boolean | null;
  latencyP50Ms: number | null;
  loads: boolean;
  maxRelDiff: number | null;
  outputMatch: boolean | null;
  provenance: string;
  runs: boolean;
  /** machine label · browser + version (+ headless) · os · @litertjs/core version */
  envLabel: string;
  /** The @litertjs/core version this record was verified against (env). */
  coreVersion: string;
}

/**
 * Deterministic one-word summary of one browser backend record. Mirrors
 * litert_compat.cards.index.browser_status (DECISIONS #62) — the generator
 * recomputes it and cross-checks against cards/index.json, so the site can
 * never disagree with the committed catalog.
 */
export function browserStatus(record: CardBrowserRecord): string {
  if (!record.loads) {
    return 'load_failed';
  }
  if (!record.runs) {
    return 'run_failed';
  }
  if (record.outputMatch === false) {
    return 'output_mismatch';
  }
  if (record.fullDelegation === false) {
    return 'fallback';
  }
  return 'pass';
}

/** The honesty gate: a demo page exists only when at least one backend ran. */
export function isDemoEligible(records: readonly CardBrowserRecord[]): boolean {
  return records.some((record) => record.runs);
}

/** One entry of a card's `device.records` array (Phase 13), as displayed. */
export interface CardDeviceRecord {
  /** Device slug (the snapshot file's <device> segment). */
  device: string;
  accelerator: string;
  date: string;
  loads: boolean;
  runs: boolean;
  outputMatch: boolean | null;
  fullDelegation: boolean | null;
  latencyP50Ms: number | null;
  decodeTokensPerS: number | null;
  provenance: string;
  /** litert (.tflite-on-NPU lane) or litert-lm (.litertlm lane). */
  runtime: string;
  /** The runtime version this record was verified against (env). */
  runtimeVersion: string;
  /** device name · soc · runtime version · vendor SDK · OS build */
  envLabel: string;
}

/**
 * Deterministic one-word summary of one device-run record. Mirrors
 * litert_compat.device_runs.records.device_status — the same vocabulary and
 * derivation as browserStatus, cross-checked against cards/index.json.
 */
export function deviceStatus(record: CardDeviceRecord): string {
  if (!record.loads) {
    return 'load_failed';
  }
  if (!record.runs) {
    return 'run_failed';
  }
  if (record.outputMatch === false) {
    return 'output_mismatch';
  }
  if (record.fullDelegation === false) {
    return 'fallback';
  }
  return 'pass';
}

/**
 * Parse an inline shape spec like '1x64:float32;1x8:int32' (mirrors
 * web/sweep/src/node/catalog.ts parseShapeSpec). Returns null for
 * fixture-file specs (*.json) or unparseable specs — those models get no
 * demo page rather than a guessed input.
 */
export function parseShapeSpec(spec: string): TensorSpec[] | null {
  if (spec.endsWith('.json')) {
    return null;
  }
  const parts = spec.split(';').map((part) => part.trim());
  const result: TensorSpec[] = [];
  for (const part of parts) {
    const match = /^(\d+(?:x\d+)*):(float32|int32)$/.exec(part);
    if (match === null) {
      return null;
    }
    const dims = match[1]!.split('x').map((dim) => Number.parseInt(dim, 10));
    result.push({ shape: dims, dtype: match[2] as 'float32' | 'int32' });
  }
  return result;
}

/**
 * The URL a demo page fetches the model from. A source_url that already
 * points at a .tflite is used as-is; otherwise it is treated as a model-repo
 * URL and the artifact is resolved Hugging-Face style (DECISIONS #66).
 */
export function resolveModelUrl(sourceUrl: string, artifactFile: string): string {
  if (sourceUrl.endsWith('.tflite')) {
    return sourceUrl;
  }
  return `${sourceUrl.replace(/\/+$/, '')}/resolve/main/${artifactFile}`;
}

export interface DelegationView {
  backend: string;
  litertVersion: string;
  coveragePct: number;
  partitions: number;
}

export interface BrowserView {
  /** backend id -> status word, recomputed and cross-checked vs index.json. */
  statuses: Record<string, string>;
  /** Repo-relative path as recorded in the card (sweep_source). */
  sweepSource: string;
  records: CardBrowserRecord[];
}

export interface DeviceView {
  records: CardDeviceRecord[];
}

export interface DemoInputs {
  inputs: TensorSpec[];
  inputSeed: number;
  warmupRuns: number;
  timedRuns: number;
}

export interface ModelView {
  id: string;
  family: string;
  task: string;
  license: string;
  sourceUrl: string;
  delegation: DelegationView | null;
  browser: BrowserView | null;
  device: DeviceView | null;
  artifactFile: string | null;
  cardJsonAbs: string;
  cardMdAbs: string;
  /** Resolved sweep_source path when it exists inside the repo. */
  sweepAbs: string | null;
  demo: DemoInputs | null;
  /** Why no demo page is generated; null when demo !== null. */
  demoSkipReason: string | null;
}

function parseEnvLabel(env: Record<string, unknown>, ctx: string): string {
  const browser = getString(env, 'browser', ctx);
  const browserVersion = getString(env, 'browser_version', ctx);
  const headless = getBoolean(env, 'headless', ctx);
  const os = getString(env, 'os', ctx);
  const machineLabel = getString(env, 'machine_label', ctx);
  const core = getString(env, 'litertjs_core_version', ctx);
  return `${machineLabel} · ${browser} ${browserVersion}${headless ? ' headless' : ''} · ${os} · @litertjs/core ${core}`;
}

function parseBrowserRecord(value: unknown, ctx: string): CardBrowserRecord {
  const record = asObject(value, ctx);
  const env = asObject(record['env'], `${ctx}.env`);
  return {
    backend: getString(record, 'backend', ctx),
    date: getString(record, 'date', ctx),
    fullDelegation: getNullableBoolean(record, 'full_delegation', ctx),
    latencyP50Ms: getNullableNumber(record, 'latency_p50_ms', ctx),
    loads: getBoolean(record, 'loads', ctx),
    maxRelDiff: getNullableNumber(record, 'max_rel_diff', ctx),
    outputMatch: getNullableBoolean(record, 'output_match', ctx),
    provenance: getString(record, 'provenance', ctx),
    runs: getBoolean(record, 'runs', ctx),
    envLabel: parseEnvLabel(env, `${ctx}.env`),
    coreVersion: getString(env, 'litertjs_core_version', `${ctx}.env`),
  };
}

function getNullableString(
  obj: Record<string, unknown>,
  key: string,
  ctx: string,
): string | null {
  const value = obj[key];
  if (value !== null && typeof value !== 'string') {
    throw new SiteDataError(`${ctx}: '${key}' must be a string or null`);
  }
  return value;
}

function parseDeviceEnvLabel(env: Record<string, unknown>, ctx: string): string {
  const parts = [getString(env, 'device', ctx)];
  const soc = getNullableString(env, 'soc', ctx);
  if (soc !== null) {
    parts.push(soc);
  }
  parts.push(`${getString(env, 'runtime', ctx)} ${getString(env, 'runtime_version', ctx)}`);
  const vendorSdk = getNullableString(env, 'vendor_sdk', ctx);
  if (vendorSdk !== null) {
    parts.push(vendorSdk);
  }
  const osBuild = getNullableString(env, 'os_build', ctx);
  if (osBuild !== null) {
    parts.push(osBuild);
  }
  return parts.join(' · ');
}

function parseDeviceRecord(value: unknown, ctx: string): CardDeviceRecord {
  const entry = asObject(value, ctx);
  const run = asObject(entry['run'], `${ctx}.run`);
  const env = asObject(run['env'], `${ctx}.run.env`);
  return {
    device: getString(entry, 'device', ctx),
    accelerator: getString(run, 'accelerator', ctx),
    date: getString(run, 'date', ctx),
    loads: getBoolean(run, 'loads', ctx),
    runs: getBoolean(run, 'runs', ctx),
    outputMatch: getNullableBoolean(run, 'output_match', ctx),
    fullDelegation: getNullableBoolean(run, 'full_delegation', ctx),
    latencyP50Ms: getNullableNumber(run, 'latency_p50_ms', ctx),
    decodeTokensPerS: getNullableNumber(run, 'decode_tokens_per_s', ctx),
    provenance: getString(run, 'provenance', ctx),
    runtime: getString(env, 'runtime', `${ctx}.run.env`),
    runtimeVersion: getString(env, 'runtime_version', `${ctx}.run.env`),
    envLabel: parseDeviceEnvLabel(env, `${ctx}.run.env`),
  };
}

interface SweepFileInfo {
  inputSpec: string;
  inputSeed: number;
  warmupRuns: number;
  timedRuns: number;
}

function readSweepFile(sweepAbs: string): SweepFileInfo {
  const ctx = sweepAbs;
  const doc = asObject(readJson(sweepAbs), ctx);
  const config = asObject(doc['config'], `${ctx}.config`);
  return {
    inputSpec: getString(doc, 'input_spec', ctx),
    inputSeed: getNumber(config, 'input_seed', `${ctx}.config`),
    warmupRuns: getNumber(config, 'warmup_runs', `${ctx}.config`),
    timedRuns: getNumber(config, 'timed_runs', `${ctx}.config`),
  };
}

function buildDemo(
  model: Pick<ModelView, 'browser' | 'artifactFile' | 'sweepAbs'>,
): { demo: DemoInputs | null; reason: string | null } {
  if (model.browser === null) {
    return { demo: null, reason: 'no browser sweep data' };
  }
  if (!isDemoEligible(model.browser.records)) {
    return { demo: null, reason: 'no backend with runs=true (honesty gate)' };
  }
  if (model.artifactFile === null) {
    return { demo: null, reason: 'card lists no artifact to load' };
  }
  if (model.sweepAbs === null) {
    return { demo: null, reason: `sweep_source not readable: ${model.browser.sweepSource}` };
  }
  const sweep = readSweepFile(model.sweepAbs);
  const inputs = parseShapeSpec(sweep.inputSpec);
  if (inputs === null) {
    return {
      demo: null,
      reason: `input_spec '${sweep.inputSpec}' is not an inline shape spec (fixture-file inputs are deferred)`,
    };
  }
  return {
    demo: {
      inputs,
      inputSeed: sweep.inputSeed,
      warmupRuns: sweep.warmupRuns,
      timedRuns: sweep.timedRuns,
    },
    reason: null,
  };
}

/** Load everything the site needs, in cards/index.json order (already sorted). */
export function loadSiteData(repoRoot: string): ModelView[] {
  const indexPath = path.join(repoRoot, 'cards', 'index.json');
  const indexDoc = asObject(readJson(indexPath), indexPath);
  const models: ModelView[] = [];
  for (const entry of getArray(indexDoc, 'models', indexPath)) {
    const indexModel = asObject(entry, `${indexPath} models[]`);
    const id = getString(indexModel, 'id', indexPath);
    const ctx = `${indexPath} model '${id}'`;

    const cardJsonAbs = path.join(repoRoot, 'cards', id, 'card.json');
    const cardMdAbs = path.join(repoRoot, 'cards', id, 'CARD.md');
    const card = asObject(readJson(cardJsonAbs), cardJsonAbs);
    const cardModel = asObject(card['model'], `${cardJsonAbs}.model`);
    if (getString(cardModel, 'id', cardJsonAbs) !== id) {
      throw new SiteDataError(`${cardJsonAbs}: model.id does not match index id '${id}'`);
    }

    const delegationRaw = card['delegation'];
    let delegation: DelegationView | null = null;
    if (delegationRaw !== null && delegationRaw !== undefined) {
      const d = asObject(delegationRaw, `${cardJsonAbs}.delegation`);
      delegation = {
        backend: getString(d, 'backend', ctx),
        litertVersion: getString(d, 'litert_version', ctx),
        coveragePct: getNumber(d, 'coverage_ops_pct', ctx),
        partitions: getNumber(d, 'partitions', ctx),
      };
    }

    const browserRaw = card['browser'];
    let browser: BrowserView | null = null;
    let sweepAbs: string | null = null;
    if (browserRaw !== null && browserRaw !== undefined) {
      const b = asObject(browserRaw, `${cardJsonAbs}.browser`);
      const records = getArray(b, 'backends', `${cardJsonAbs}.browser`).map((r, i) =>
        parseBrowserRecord(r, `${cardJsonAbs}.browser.backends[${String(i)}]`),
      );
      const statuses: Record<string, string> = {};
      for (const record of records) {
        statuses[record.backend] = browserStatus(record);
      }
      // Cross-check against the committed index.json statuses.
      const indexBrowser = asObject(indexModel['browser'], `${ctx}.browser`);
      const indexStatuses = asObject(indexBrowser['statuses'], `${ctx}.browser.statuses`);
      const want = JSON.stringify(
        Object.fromEntries(Object.entries(statuses).sort(([a], [b2]) => (a < b2 ? -1 : 1))),
      );
      const got = JSON.stringify(
        Object.fromEntries(Object.entries(indexStatuses).sort(([a], [b2]) => (a < b2 ? -1 : 1))),
      );
      if (want !== got) {
        throw new SiteDataError(
          `${ctx}: browser statuses in index.json (${got}) disagree with card records (${want}) — regenerate the index`,
        );
      }
      const sweepSource = getString(b, 'sweep_source', `${cardJsonAbs}.browser`);
      const candidate = path.resolve(repoRoot, sweepSource);
      if (candidate.startsWith(repoRoot + path.sep) && fs.existsSync(candidate)) {
        sweepAbs = candidate;
      }
      browser = { statuses, sweepSource, records };
    }

    const deviceRaw = card['device'];
    let device: DeviceView | null = null;
    if (deviceRaw !== null && deviceRaw !== undefined) {
      const dev = asObject(deviceRaw, `${cardJsonAbs}.device`);
      const records = getArray(dev, 'records', `${cardJsonAbs}.device`).map((r, i) =>
        parseDeviceRecord(r, `${cardJsonAbs}.device.records[${String(i)}]`),
      );
      // Cross-check against the committed index.json device statuses.
      const indexDevice = asObject(indexModel['device'], `${ctx}.device`);
      const want = JSON.stringify(
        records.map((r) => [r.device, r.accelerator, deviceStatus(r)]),
      );
      const got = JSON.stringify(
        getArray(indexDevice, 'records', `${ctx}.device`).map((r) => {
          const row = asObject(r, `${ctx}.device.records[]`);
          return [
            getString(row, 'device', ctx),
            getString(row, 'accelerator', ctx),
            getString(row, 'status', ctx),
          ];
        }),
      );
      if (want !== got) {
        throw new SiteDataError(
          `${ctx}: device statuses in index.json (${got}) disagree with card records (${want}) — regenerate the index`,
        );
      }
      device = { records };
    }

    const artifacts = getArray(card, 'artifacts', cardJsonAbs);
    let artifactFile: string | null = null;
    if (artifacts.length > 0) {
      artifactFile = getString(asObject(artifacts[0], ctx), 'file', ctx);
    }

    const partial = { browser, artifactFile, sweepAbs };
    const { demo, reason } = buildDemo(partial);
    models.push({
      id,
      family: getString(cardModel, 'family', cardJsonAbs),
      task: getString(cardModel, 'task', cardJsonAbs),
      license: getString(cardModel, 'license', cardJsonAbs),
      sourceUrl: getString(cardModel, 'source_url', cardJsonAbs),
      delegation,
      browser,
      device,
      artifactFile,
      cardJsonAbs,
      cardMdAbs,
      sweepAbs,
      demo,
      demoSkipReason: reason,
    });
  }
  return models;
}
