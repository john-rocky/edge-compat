/**
 * Staleness surfacing (Phase 11): compare a verified-against version to the
 * latest KNOWN upstream release from data/releases.json.
 *
 * Mirrors litert_compat.freshness.releases semantics exactly: numeric-prefix
 * comparison, strictly-numeric verified-against versions only (a placeholder
 * like '0.0.0-example' makes no claim about a real runtime, so it can never
 * lag one), and no guessing — unparsable versions are never flagged.
 * Staleness changes NO verdict and NO data; it only adds a visible flag.
 */

export interface LatestKnownRelease {
  version: string;
  checkedAt: string;
}

/** Numeric-prefix version tuple; null when no leading numeric component. */
export function parseVersion(version: string): number[] | null {
  const parts: number[] = [];
  for (const component of version.split('.')) {
    const match = /^(\d+)/.exec(component);
    if (match === null) {
      break;
    }
    parts.push(Number.parseInt(match[1]!, 10));
    if (match[1] !== component) {
      break;
    }
  }
  return parts.length > 0 ? parts : null;
}

/**
 * True when `version` is strictly older than `latest`. `version` must be
 * strictly numeric (x.y.z); ties and unparsable versions are never stale.
 */
export function lagsBehind(version: string, latest: string): boolean {
  if (!/^\d+(\.\d+)*$/.test(version)) {
    return false;
  }
  const a = parseVersion(version);
  const b = parseVersion(latest);
  if (a === null || b === null) {
    return false;
  }
  const width = Math.max(a.length, b.length);
  for (let i = 0; i < width; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    if (x !== y) {
      return x < y;
    }
  }
  return false;
}

/**
 * The latest known release when `version` lags it, else null. This is the
 * one predicate every stale flag in the site goes through.
 */
export function staleAgainst(
  version: string,
  latest: LatestKnownRelease | null,
): LatestKnownRelease | null {
  if (latest === null || !lagsBehind(version, latest.version)) {
    return null;
  }
  return latest;
}
