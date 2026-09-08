/**
 * Deterministic HTML rendering: same inputs => same bytes. No timestamps, no
 * randomness, sorted iteration everywhere. Failing statuses render exactly
 * like passing ones — a model that does not run in the browser is
 * information, not an embarrassment to hide.
 */
import { DISCLOSURE_LINE, NAMING_NOTICE, PROJECT_NAME } from '../shared/branding.ts';
import type { DemoConfig } from '../shared/config.ts';
import type { PlaygroundConfig } from '../shared/playground.ts';
import { type LatestKnownRelease, staleAgainst } from '../shared/staleness.ts';
import { deviceStatus, type ModelView } from './data.ts';

export function escapeHtml(text: string): string {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

/** Sort rank per status word — lower is better; unmeasured sorts last. */
const STATUS_RANK: Record<string, number> = {
  pass: 0,
  fallback: 1,
  output_mismatch: 2,
  run_failed: 3,
  load_failed: 4,
};

const HONESTY_NOTE =
  'Every number is labeled with the environment that produced it — ' +
  'headless desktop Chromium is not a user’s phone, and results are ' +
  'only meaningful together with their environment.';

/**
 * Every public page carries the disclosure line and the affiliation
 * notice (Phase 12.1) — single-sourced in shared/branding.ts, held in
 * parity with the Python constants by a site test, and verified on built
 * output by `edge-compat release-gate --site-dist`.
 */
const SITE_FOOTER =
  `<footer><p>${escapeHtml(DISCLOSURE_LINE)}</p>` +
  `<p>${escapeHtml(PROJECT_NAME)} demo zoo — ${escapeHtml(NAMING_NOTICE)} ` +
  'Generated deterministically from <code>cards/index.json</code> and per-model ' +
  '<code>card.json</code>; model weights are never hosted here.</p></footer>';

/**
 * The launch plan's mandatory "reproduce this with an agent" block
 * (Phase 12.2): a one-line prompt for Claude Code / Gemini CLI naming real
 * commands only, with a copy button wired by assets/copy.js.
 */
function reproSection(intro: string, prompt: string): string {
  return (
    '<h2>Reproduce this with an agent</h2>\n<div class="cta">\n' +
    `<p>${escapeHtml(intro)}</p>\n` +
    `<pre id="repro-prompt">${escapeHtml(prompt)}</pre>\n` +
    '<button data-copy-target="repro-prompt">Copy prompt</button>\n</div>\n' +
    '\n'
  );
}

export const INDEX_REPRODUCE_PROMPT =
  `In the ${PROJECT_NAME} repo, re-run the LiteRT.js sweep behind this table on the ` +
  'committed example catalog: cd web/sweep && npm ci && npm run sweep -- ' +
  '--catalog ../../data/examples/web_catalog_example.csv --out out/ — then compare ' +
  'the records under out/ with the example rows in this table.';

function pageShell(title: string, cssHref: string, body: string, scripts: string): string {
  return (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' +
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n' +
    `<title>${escapeHtml(title)}</title>\n` +
    `<link rel="stylesheet" href="${escapeHtml(cssHref)}">\n` +
    '</head>\n<body>\n' +
    body +
    '\n' +
    scripts +
    '\n</body>\n</html>\n'
  );
}

/** Backend columns: union of backends present in the data, sorted. */
export function backendColumns(models: readonly ModelView[]): string[] {
  const backends = new Set<string>();
  for (const model of models) {
    for (const backend of Object.keys(model.browser?.statuses ?? {})) {
      backends.add(backend);
    }
  }
  return [...backends].sort();
}

/**
 * The visible stale flag (Phase 11): shown whenever a verified-against
 * @litertjs/core version lags the latest known release. Display only — the
 * measured values and statuses next to it are unchanged.
 */
function staleFlag(
  stale: { version: string; checkedAt: string } | null,
  runtimeLabel = '@litertjs/core',
): string {
  if (stale === null) {
    return '';
  }
  return (
    ` <span class="stale-flag" title="Verified against an older ${escapeHtml(runtimeLabel)} than the ` +
    `latest known release ${escapeHtml(stale.version)} (checked ${escapeHtml(stale.checkedAt)}). ` +
    `The measurements are unchanged — they are just older than the current release.">stale — ` +
    `latest known ${escapeHtml(stale.version)}</span>`
  );
}

/** Latest known releases for the device-run runtime axes (data/releases.json). */
export interface DeviceReleases {
  litert: LatestKnownRelease | null;
  litertlm: LatestKnownRelease | null;
}

const NO_DEVICE_RELEASES: DeviceReleases = { litert: null, litertlm: null };

/**
 * The "Device runs" cell: one line per (device x accelerator) record —
 * status, headline number, stale flag against the record's own runtime axis
 * (litert vs litert-lm), and the env label. Mirrors the browser cells'
 * honesty rules: env always shown, decay flagged, never hidden.
 */
function deviceCell(model: ModelView, deviceReleases: DeviceReleases): string {
  const records = model.device?.records ?? [];
  if (records.length === 0) {
    return '<td data-sort-value="9" class="status-not_measured">not measured</td>';
  }
  const worst = Math.max(...records.map((r) => STATUS_RANK[deviceStatus(r)] ?? 8));
  const lines = records.map((record) => {
    const status = deviceStatus(record);
    const headline =
      record.decodeTokensPerS !== null
        ? ` · ${String(record.decodeTokensPerS)} tok/s decode`
        : record.latencyP50Ms !== null
          ? ` · ${String(record.latencyP50Ms)} ms`
          : '';
    const provenance =
      record.provenance === 'example'
        ? ' <span class="provenance-example">example</span>'
        : '';
    const latest =
      record.runtime === 'litert-lm' ? deviceReleases.litertlm : deviceReleases.litert;
    return (
      `<span class="device-run"><code>${escapeHtml(record.device)}</code> ` +
      `${escapeHtml(record.accelerator)}: ` +
      `<span class="status-${escapeHtml(status)}">${escapeHtml(status)}</span>${headline}${provenance}` +
      staleFlag(staleAgainst(record.runtimeVersion, latest), record.runtime) +
      `<span class="env">${escapeHtml(record.envLabel)} · ${escapeHtml(record.date)}</span></span>`
    );
  });
  return `<td data-sort-value="${String(worst)}">${lines.join('<br>')}</td>`;
}

function backendCell(
  model: ModelView,
  backend: string,
  latestKnown: LatestKnownRelease | null,
): string {
  const record = model.browser?.records.find((r) => r.backend === backend);
  if (record === undefined) {
    return '<td data-sort-value="9" class="status-not_measured">not measured</td>';
  }
  const status = model.browser!.statuses[backend]!;
  const rank = STATUS_RANK[status] ?? 8;
  const latency =
    record.latencyP50Ms === null ? '' : ` · ${String(record.latencyP50Ms)} ms`;
  const provenance =
    record.provenance === 'example'
      ? ' <span class="provenance-example">example</span>'
      : '';
  return (
    `<td data-sort-value="${String(rank)}">` +
    `<span class="status-${escapeHtml(status)}">${escapeHtml(status)}</span>${latency}${provenance}` +
    staleFlag(staleAgainst(record.coreVersion, latestKnown)) +
    `<span class="env">${escapeHtml(record.envLabel)} · ${escapeHtml(record.date)}</span>` +
    '</td>'
  );
}

export function renderIndexPage(
  models: readonly ModelView[],
  latestKnown: LatestKnownRelease | null = null,
  deviceReleases: DeviceReleases = NO_DEVICE_RELEASES,
): string {
  const backends = backendColumns(models);
  const anyDevice = models.some((m) => (m.device?.records ?? []).length > 0);
  const anyExample = models.some(
    (m) =>
      (m.browser?.records ?? []).some((r) => r.provenance === 'example') ||
      (m.device?.records ?? []).some((r) => r.provenance === 'example'),
  );
  const allExample =
    anyExample &&
    models.every(
      (m) =>
        (m.browser?.records ?? []).every((r) => r.provenance === 'example') &&
        (m.device?.records ?? []).every((r) => r.provenance === 'example'),
    );

  const header = ['<th aria-sort="ascending">Model</th>', '<th>Task</th>', '<th>Native delegation</th>'];
  for (const backend of backends) {
    header.push(`<th data-sort-type="number">${escapeHtml(backend)}</th>`);
  }
  if (anyDevice) {
    header.push('<th data-sort-type="number">Device runs</th>');
  }
  header.push('<th>Card</th>', '<th>Demo</th>');

  const rows: string[] = [];
  for (const model of models) {
    const cells: string[] = [];
    cells.push(`<td><code>${escapeHtml(model.id)}</code></td>`);
    cells.push(`<td>${escapeHtml(model.task)}</td>`);
    const d = model.delegation;
    cells.push(
      d === null
        ? '<td class="status-not_measured">—</td>'
        : `<td>${escapeHtml(d.backend)} @ litert ${escapeHtml(d.litertVersion)}: ` +
            `${String(d.coveragePct)}% · ${String(d.partitions)} partition(s)</td>`,
    );
    for (const backend of backends) {
      cells.push(backendCell(model, backend, latestKnown));
    }
    if (anyDevice) {
      cells.push(deviceCell(model, deviceReleases));
    }
    cells.push(
      `<td><a href="cards/${escapeHtml(model.id)}/CARD.md">CARD.md</a> · ` +
        `<a href="cards/${escapeHtml(model.id)}/card.json">card.json</a></td>`,
    );
    cells.push(
      model.demo !== null
        ? `<td><a href="models/${escapeHtml(model.id)}/">run it</a></td>`
        : `<td class="status-not_measured" title="${escapeHtml(model.demoSkipReason ?? '')}">—</td>`,
    );
    rows.push(`<tr>${cells.join('')}</tr>`);
  }

  const banner = anyExample
    ? '<div class="banner">' +
      (allExample
        ? 'All browser and device results on this page are <b>example-provenance pipeline fixtures</b> — not real model measurements. Real measured data replaces them as it lands.'
        : 'Rows marked <b>example</b> are pipeline fixtures, not real model measurements.') +
      '</div>'
    : '';

  const body =
    '<h1>LiteRT.js Demo Zoo</h1>\n' +
    '<p class="subtitle">Measured browser compatibility for LiteRT (.tflite) models: ' +
    'what loads, what delegates to WebGPU, what matches CPU outputs, and at what latency. ' +
    'Passing <em>and failing</em> models are listed — a model that does not run in the ' +
    'browser is information.</p>\n' +
    '<p class="check-cta"><a href="check/">Check <em>your</em> model → drop a .tflite ' +
    'and see if it runs, right here in your browser. Nothing is uploaded.</a></p>\n' +
    banner +
    `<p class="note">${HONESTY_NOTE} Click a column header to sort.</p>\n` +
    '<div class="table-wrap">\n<table class="sortable">\n<thead>\n' +
    `<tr>${header.join('')}</tr>\n` +
    '</thead>\n<tbody>\n' +
    rows.join('\n') +
    '\n</tbody>\n</table>\n</div>\n' +
    reproSection(
      'One-line prompt for Claude Code / Gemini CLI — re-runs the sweep behind ' +
        'this table on the committed example catalog, locally:',
      INDEX_REPRODUCE_PROMPT,
    ) +
    SITE_FOOTER;

  return pageShell(
    'LiteRT.js Demo Zoo — measured browser compatibility',
    'assets/site.css',
    body,
    '<script src="assets/sort.js" defer></script>\n' +
      '<script src="assets/copy.js" defer></script>',
  );
}

function sweepTable(config: DemoConfig): string {
  const rows = config.sweep.map((record) => {
    const cell = (value: boolean | null, yes: string, no: string): string =>
      value === null ? '—' : value ? yes : no;
    return (
      '<tr>' +
      `<td><code>${escapeHtml(record.backend)}</code></td>` +
      `<td><span class="status-${escapeHtml(record.status)}">${escapeHtml(record.status)}</span></td>` +
      `<td>${cell(record.fullDelegation, 'yes', 'no')}</td>` +
      `<td>${cell(record.outputMatch, 'pass', 'fail')}</td>` +
      `<td>${record.latencyP50Ms === null ? '—' : String(record.latencyP50Ms)}</td>` +
      `<td>${escapeHtml(record.envLabel)}</td>` +
      `<td>${escapeHtml(record.date)}${staleFlag(record.staleAgainst)}</td>` +
      `<td>${escapeHtml(record.provenance)}</td>` +
      '</tr>'
    );
  });
  return (
    '<div class="table-wrap">\n<table>\n<thead>\n' +
    '<tr><th>Backend</th><th>Status</th><th>Full delegation</th><th>Output match</th>' +
    '<th>Latency p50 (ms)</th><th>Environment</th><th>Verified</th><th>Provenance</th></tr>\n' +
    '</thead>\n<tbody>\n' +
    rows.join('\n') +
    '\n</tbody>\n</table>\n</div>'
  );
}

/** Serialize an embedded config block; escape to keep </script> inert. */
function embedConfig(config: DemoConfig | PlaygroundConfig): string {
  return JSON.stringify(config).replaceAll('<', '\\u003c');
}

export function renderDemoPage(model: ModelView, config: DemoConfig): string {
  const exampleBanner = config.sweep.some((r) => r.provenance === 'example')
    ? '<div class="banner">This is an <b>example-provenance pipeline fixture</b> — a synthetic model exercising the demo pipeline, not a real published model.</div>\n'
    : '';
  const ctaLinks: string[] = [];
  if (config.cookbookUrl !== null) {
    ctaLinks.push(`<a href="${escapeHtml(config.cookbookUrl)}">litert-cookbook</a>`);
  }
  if (config.repoUrl !== null) {
    ctaLinks.push(`<a href="${escapeHtml(config.repoUrl)}">repository</a>`);
  }

  const body =
    `<p><a href="../../">← all models</a></p>\n` +
    `<h1><code>${escapeHtml(model.id)}</code></h1>\n` +
    `<p class="subtitle">${escapeHtml(model.task)} · family: ${escapeHtml(model.family)}</p>\n` +
    exampleBanner +
    '<div class="table-wrap">\n<table>\n<tbody>\n' +
    `<tr><th>Source</th><td><a href="${escapeHtml(model.sourceUrl)}">${escapeHtml(model.sourceUrl)}</a></td></tr>\n` +
    `<tr><th>License</th><td>${escapeHtml(model.license)}</td></tr>\n` +
    `<tr><th>Model file</th><td><code>${escapeHtml(model.artifactFile ?? '')}</code> — downloaded from the source at runtime; this site hosts no weights</td></tr>\n` +
    '</tbody>\n</table>\n</div>\n' +
    '<h2>Run it in your browser</h2>\n' +
    '<div class="live-panel">\n' +
    '<button id="run-webgpu">Run on WebGPU</button>\n' +
    '<button id="run-wasm" class="secondary">Run on WASM (CPU)</button>\n' +
    '<div class="metrics">\n' +
    '<div class="metric"><b id="live-backend">—</b><span>backend actually in use</span></div>\n' +
    '<div class="metric"><b id="live-latency">—</b><span>live latency p50</span></div>\n' +
    '</div>\n' +
    '<p id="live-status">Preparing…</p>\n' +
    '</div>\n' +
    '<h2>Measured sweep results</h2>\n' +
    `<p class="note">${HONESTY_NOTE}</p>\n` +
    sweepTable(config) +
    `<p class="note">Full record (failure class, evidence, sweep config): ` +
    `<a href="../../sweeps/${escapeHtml(model.id)}.json">raw sweep JSON</a> · ` +
    `<a href="../../cards/${escapeHtml(model.id)}/card.json">card.json</a> · ` +
    `<a href="../../cards/${escapeHtml(model.id)}/CARD.md">CARD.md</a></p>\n` +
    '<h2>Build this yourself</h2>\n' +
    '<div class="cta">\n' +
    '<p>One-line prompt for your AI coding assistant (uses the litert-cookbook web skill):</p>\n' +
    `<pre id="cta-prompt">${escapeHtml(config.ctaPrompt)}</pre>\n` +
    '<button id="cta-copy">Copy prompt</button>' +
    (ctaLinks.length > 0 ? `\n<p class="note">${ctaLinks.join(' · ')}</p>` : '') +
    '\n</div>\n' +
    reproSection(
      'One-line prompt for Claude Code / Gemini CLI — re-runs the same sweep ' +
        'verification for this model, locally:',
      config.reproPrompt,
    ) +
    SITE_FOOTER;

  return pageShell(
    `${model.id} — LiteRT.js demo`,
    '../../assets/site.css',
    body,
    `<script type="application/json" id="demo-config">${embedConfig(config)}</script>\n` +
      '<script src="../../assets/demo.js" defer></script>\n' +
      '<script src="../../assets/copy.js" defer></script>',
  );
}

function snapshotTable(refs: PlaygroundConfig['matrix']): string {
  if (refs.length === 0) {
    return (
      '<p class="note">No web-backend matrix snapshots are exported yet, so every op ' +
      'below will honestly read <code>unknown</code>. Op-level browser entries are ' +
      'created only when a runtime log names a specific op, with the log captured as ' +
      'evidence — never derived from model-level results.</p>'
    );
  }
  const rows = refs.map(
    (s) =>
      '<tr>' +
      `<td><code>${escapeHtml(s.backend)}</code></td>` +
      `<td>${escapeHtml(s.litertVersion)}${staleFlag(s.staleAgainst)}</td>` +
      `<td>${escapeHtml(s.generatedAt)}</td>` +
      `<td>${String(s.entryCount)}</td>` +
      `<td><a href="../matrix/${escapeHtml(s.backend)}.json">raw JSON</a></td>` +
      '</tr>',
  );
  return (
    '<div class="table-wrap">\n<table>\n<thead>\n' +
    '<tr><th>Backend</th><th>Verified against @litertjs/core</th><th>Generated</th>' +
    '<th>Entries</th><th>Snapshot</th></tr>\n' +
    '</thead>\n<tbody>\n' +
    rows.join('\n') +
    '\n</tbody>\n</table>\n</div>'
  );
}

export function renderPlaygroundPage(config: PlaygroundConfig): string {
  const body =
    `<p><a href="../">← all models</a></p>\n` +
    '<h1>Check your model</h1>\n' +
    '<p class="subtitle">Does your .tflite run in the browser? Drop it in and find out ' +
    'in seconds — it loads, compiles, and runs right here, on your hardware.</p>\n' +
    '<div class="banner privacy-banner"><b>Your model never leaves your browser.</b> ' +
    'The file is read and run locally by LiteRT.js inside this page. Nothing is ' +
    'uploaded, and no network request contains your model or anything derived from ' +
    'it — there is no server to send it to. That is the point of this page.</div>\n' +
    '<div class="drop-zone" id="drop-zone">\n' +
    '<p><b>Drop a <code>.tflite</code> file here</b> — or pick one:</p>\n' +
    '<input type="file" id="model-input" accept=".tflite,application/octet-stream">\n' +
    '</div>\n' +
    '<p id="pg-status">Preparing…</p>\n' +
    '<div id="results-section" hidden>\n' +
    '<h2>Live results — your browser, your hardware</h2>\n' +
    `<p class="note">${HONESTY_NOTE} The numbers below were measured just now in ` +
    '<em>your</em> environment (shown underneath) — they are not the reference ' +
    'measurements from the zoo table.</p>\n' +
    '<div class="table-wrap">\n<table>\n<thead>\n' +
    '<tr><th>Backend</th><th>Loads</th><th>Runs</th><th>Full delegation</th>' +
    '<th>Output match</th><th>Latency p50 (ms)</th><th>Failure</th></tr>\n' +
    '</thead>\n<tbody id="results-body">\n</tbody>\n</table>\n</div>\n' +
    '<p class="note" id="env-block"></p>\n' +
    '<details id="evidence-details" hidden><summary>Runtime console messages ' +
    '(delegation evidence, verbatim)</summary><pre id="evidence-pre"></pre></details>\n' +
    '</div>\n' +
    '<div id="xref-section" hidden>\n' +
    '<h2>Ops × compatibility matrix</h2>\n' +
    '<p class="note">The operator inventory is parsed from your model locally ' +
    '(operator names only — full graph analysis is what <code>edge-lint</code> does). ' +
    'Statuses come <b>only</b> from the exported matrix snapshots listed below — real ' +
    'measurements, never inferred by this page. Ops without a matrix entry read ' +
    '<code>unknown</code>, and honestly so: op-level browser data exists only where a ' +
    'runtime log named the op.</p>\n' +
    '<div class="table-wrap">\n<table>\n<thead>\n<tr id="xref-head"></tr>\n</thead>\n' +
    '<tbody id="xref-body">\n</tbody>\n</table>\n</div>\n' +
    '</div>\n' +
    '<h2>Exported matrix snapshots</h2>\n' +
    snapshotTable(config.matrix) +
    '\n<div id="cta-section" hidden>\n' +
    '<h2>Next steps</h2>\n' +
    '<div class="cta">\n' +
    '<p>Full partition analysis (partition map, blocking-op ranking, rewrite hints) ' +
    'runs locally with <code>edge-lint</code> — your model stays on your machine ' +
    'there too:</p>\n' +
    '<pre id="lint-command"></pre>\n' +
    '<button id="lint-copy">Copy command</button>\n' +
    '<p>One-line prompt for your AI coding assistant (uses the litert-cookbook web ' +
    'skill):</p>\n' +
    '<pre id="cta-prompt"></pre>\n' +
    '<button id="cta-copy">Copy prompt</button>\n' +
    '</div>\n</div>\n' +
    reproSection(
      'One-line prompt for Claude Code / Gemini CLI — the same static verdicts ' +
        'locally with full partition analysis (your model stays on your machine):',
      config.reproPrompt,
    ) +
    SITE_FOOTER;

  return pageShell(
    'Check your model — LiteRT.js playground',
    '../assets/site.css',
    body,
    `<script type="application/json" id="playground-config">${embedConfig(config)}</script>\n` +
      '<script src="../assets/playground.js" defer></script>\n' +
      '<script src="../assets/copy.js" defer></script>',
  );
}
