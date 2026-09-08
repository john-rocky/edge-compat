/**
 * Single-source naming + disclosure line (Phase 12) — mirrors
 * src/litert_compat/branding.py. A site test parses the Python source and
 * asserts exact parity, so a rename or a disclosure-wording change is one
 * edit per lane and nothing can drift. Edit the Python side first.
 */

export const PROJECT_NAME = 'edge-compat';

/** Rendered on every public page; checked by the release gate. */
export const DISCLOSURE_LINE = 'Built by john-rocky. Measurements and views are my own.';

/** The affiliation notice: an unofficial project, not a Google product. */
export const NAMING_NOTICE =
  'edge-compat is an unofficial project for LiteRT, not an official Google product.';
