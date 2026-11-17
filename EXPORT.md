# Public export

Exported on 2026-11-17 from the private lab repository at commit `6b4bad9dfb1f842db69ae35f5e2ab6d37dd35ea3`
by `tools/export_public.sh` (one squashed commit per export; the lab's
history is not published).

Scrubbed: the local home directory in evidence paths (`/Users/USER`).
Not exported: process logs (`DECISIONS.md`, `PROGRESS.md`), the spec and
release runbook, launch drafts, staging data under `data/*_staging/`, the
regression run data (`data/regression/`) and the machine-local path files of
the regression tools, two notes kept beside the LLM log samples, the export's
pattern lists (`tools/export_guard/`), and the lab-only workflows (resweep,
autobump, agentfix, reminders, export).
Nothing else is edited: code, schemas, matrix / sweep / device-run snapshots,
cards, the site, the MCP server and the tests are verbatim.
