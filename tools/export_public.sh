#!/usr/bin/env bash
# Export a scrubbed snapshot of this lab repository into the public mirror
# (github.com/john-rocky/edge-compat) as one squashed commit.
#
#   tools/export_public.sh <mirror-clone-dir> [--push]
#
# What stays private: the process logs (DECISIONS.md, PROGRESS.md), the spec
# and the release runbook, launch drafts, staging data, and the workflows that
# commit back into the lab (resweep, autobump, agentfix, reminders, this
# export). What is scrubbed: the local home directory in evidence paths
# (/Users/USER). Everything else — code, schemas, matrix + sweep + device-run
# data, cards, the site, the MCP server, tests — is exported verbatim, and the
# mirror's EXPORT.md names the source commit. The export is the message: no
# history is rewritten and no measurement is edited. Before the mirror is
# touched, the whole stage is read for text that must not go out (the lab-only
# tools/export_guard); a hit, or a missing guard, stops the export.
set -euo pipefail

SRC="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
DEST="${1:?usage: tools/export_public.sh <mirror-clone-dir> [--push]}"
PUSH="${2:-}"
SHA="$(git -C "$SRC" rev-parse HEAD)"
SHORT="${SHA:0:7}"
DATE="$(date -u +%Y-%m-%d)"

PRIVATE_ONLY=(
  DECISIONS.md PROGRESS.md litert-compat-spec.md RELEASE.md scoreboard.md
  launch handover
  data/bench_staging data/cards_staging data/device_runs_staging data/matrix_staging
  data/meta_staging data/transforms_staging
  data/regression tools/regression/quiet_wait.path tools/regression/manifests.path
  data/examples/llm_samples/README.md data/examples/llm_samples/gpu_audit/MATRIX.excerpt.md
  tools/export_guard
  .github/workflows/agentfix.yml .github/workflows/autobump.yml
  .github/workflows/resweep.yml .github/workflows/reminders.yml
  .github/workflows/export-public.yml
)
SCRUB_FROM='/Users/USER'
SCRUB_TO='/Users/USER'

if [ ! -d "$DEST/.git" ]; then
  echo "error: $DEST is not a git clone of the mirror" >&2
  exit 2
fi
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
  echo "error: lab working tree has uncommitted tracked changes; export from a clean HEAD" >&2
  exit 2
fi

STAGE="$(mktemp -d)"
LIST="$(mktemp)"
trap 'rm -rf "$STAGE" "$LIST"' EXIT
git -C "$SRC" archive --format=tar HEAD | tar -x -C "$STAGE"
for p in "${PRIVATE_ONLY[@]}"; do rm -rf "${STAGE:?}/$p"; done
# Every data/*_staging directory is private, listed above or not: data/device_runs_staging
# was created on 2026-09-28, missed the list, and went out with the 2026-10-01 export.
for d in "${STAGE:?}"/data/*_staging; do
  if [ -d "$d" ]; then rm -rf "$d"; fi
done

# Scrub the home directory from every text file; refuse to export if any
# occurrence survives (a binary would be the only way, and there is none).
grep -rIl --exclude-dir=.git -- "$SCRUB_FROM" "$STAGE" | while read -r f; do
  perl -pi -e "s|\Q$SCRUB_FROM\E|$SCRUB_TO|g" "$f"
done
if grep -rIq --exclude-dir=.git -- "$(basename "$SCRUB_FROM")" "$STAGE"; then
  echo "error: local username still present after scrub" >&2
  grep -rIl --exclude-dir=.git -- "$(basename "$SCRUB_FROM")" "$STAGE" >&2
  exit 3
fi

cat > "$STAGE/EXPORT.md" <<MD
# Public export

Exported on $DATE from the private lab repository at commit \`$SHA\`
by \`tools/export_public.sh\` (one squashed commit per export; the lab's
history is not published).

Scrubbed: the local home directory in evidence paths (\`/Users/USER\`).
Not exported: process logs (\`DECISIONS.md\`, \`PROGRESS.md\`), the spec and
release runbook, launch drafts, staging data under \`data/*_staging/\`, the
regression run data (\`data/regression/\`) and the machine-local path files of
the regression tools, two notes kept beside the LLM log samples, the export's
pattern lists (\`tools/export_guard/\`), and the lab-only workflows (resweep,
autobump, agentfix, reminders, export).
Nothing else is edited: code, schemas, matrix / sweep / device-run snapshots,
cards, the site, the MCP server and the tests are verbatim.
MD

# Read the whole stage, EXPORT.md included, for text that must not go out:
# every pattern of tools/export_guard/patterns.tsv (one Python regex per row,
# id<TAB>regex) and every line that tools/export_guard/check reports (a text
# checker run over the stage). Both live only in the lab; a missing guard, an
# empty pattern list or any hit stops the export before the mirror is touched.
GUARD="$SRC/tools/export_guard"
if [ ! -f "$GUARD/patterns.tsv" ] || [ ! -x "$GUARD/check" ]; then
  echo "error: $GUARD (patterns.tsv, check) is missing; export refused" >&2
  exit 4
fi
if ! python3 -I - "$GUARD/patterns.tsv" "$STAGE" <<'PY'
import os, re, sys
pats = []
for row in open(sys.argv[1], encoding="utf-8"):
    if row.startswith("#") or not row.strip():
        continue
    pid, rx = row.rstrip("\n").split("\t")[:2]
    pats.append((pid, re.compile(rx)))
root, hits = sys.argv[2], 0
for dp, dn, fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        data = open(p, "rb").read()
        if b"\0" in data[:8192]:
            continue
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for pid, rx in pats:
                if rx.search(line):
                    hits += 1
                    print(f"  {os.path.relpath(p, root)}:{n}: {pid}", file=sys.stderr)
print(f"export guard: {len(pats)} pattern(s), {hits} hit(s)", file=sys.stderr)
sys.exit(1 if hits or not pats else 0)
PY
then
  echo "error: the stage holds text that must not go out (above); export refused" >&2
  exit 4
fi
if ! "$GUARD/check" "$STAGE" "$GUARD/reviewed.tsv"; then
  echo "error: the text check of the stage did not pass (above); export refused" >&2
  exit 4
fi

# Replace the mirror's tree with the staged export (deletions included), then
# stage exactly the exported files: tracked ones by update, the rest by name,
# so nothing else lying in the mirror clone is committed.
(cd "$DEST" && git ls-files -z | xargs -0 rm -f && find . -mindepth 1 -type d -empty -not -path './.git*' -delete)
cp -R "$STAGE"/. "$DEST"/
(cd "$STAGE" && find . -type f -print0) > "$LIST"
(cd "$DEST" && git add -u && \
  GIT_LITERAL_PATHSPECS=1 git add --pathspec-from-file="$LIST" --pathspec-file-nul && \
  if git diff --cached --quiet; then echo "mirror already at $SHORT: nothing to export"; exit 0; fi && \
  git commit -q -m "export: $DATE from lab $SHORT" && git log -1 --format='%h %s')
if [ "$PUSH" = "--push" ]; then
  git -C "$DEST" push origin HEAD:main
fi
