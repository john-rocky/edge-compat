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
# history is rewritten and no measurement is edited.
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
  data/bench_staging data/cards_staging data/matrix_staging data/meta_staging
  data/transforms_staging
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
trap 'rm -rf "$STAGE"' EXIT
git -C "$SRC" archive --format=tar HEAD | tar -x -C "$STAGE"
for p in "${PRIVATE_ONLY[@]}"; do rm -rf "${STAGE:?}/$p"; done

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
release runbook, launch drafts, staging data under \`data/*_staging/\`, and the
lab-only workflows (resweep, autobump, agentfix, reminders, export).
Nothing else is edited: code, schemas, matrix / sweep / device-run snapshots,
cards, the site, the MCP server and the tests are verbatim.
MD

# Replace the mirror's tree with the staged export (deletions included).
(cd "$DEST" && git ls-files -z | xargs -0 rm -f && find . -mindepth 1 -type d -empty -not -path './.git*' -delete)
cp -R "$STAGE"/. "$DEST"/
(cd "$DEST" && git add -A && \
  if git diff --cached --quiet; then echo "mirror already at $SHORT: nothing to export"; exit 0; fi && \
  git commit -q -m "export: $DATE from lab $SHORT" && git log -1 --format='%h %s')
if [ "$PUSH" = "--push" ]; then
  git -C "$DEST" push origin HEAD:main
fi
