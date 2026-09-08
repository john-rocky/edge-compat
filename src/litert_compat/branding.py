"""Single-source project naming + disclosure line (Phase 12).

The public name and the disclosure wording are each one edit here — mirrored
in site/src/shared/branding.ts, which a site test holds in parity with this
file — followed by regenerating the committed outputs (`edge-card index
cards`). No string chasing across templates: every public surface renders
these constants, and `edge-compat release-gate` verifies their presence
before anything goes public.
"""

from __future__ import annotations

# Public name (settled 2026-09-08). "LiteRT" stays out of the project and CLI
# names — Google's trademark guidelines allow descriptive use ("for LiteRT")
# but not a Google brand feature inside a third-party product name.
PROJECT_NAME = "edge-compat"
LINT_TOOL_NAME = "edge-lint"

# Rendered on every public surface (site footer, cards/README.md, llms.txt,
# README.md) and checked by the release gate.
DISCLOSURE_LINE = "Built by john-rocky. Measurements and views are my own."

# The affiliation notice: an unofficial project, not a Google product.
NAMING_NOTICE = (
    "edge-compat is an unofficial project for LiteRT, not an official Google product."
)
