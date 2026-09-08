"""Freshness system (Phase 11): keep browser-lane data current where automation
honestly can, and turn what it cannot automate into scheduled reminders.

Hard rule (spec Phase 11 Out): nothing in this package writes op-level matrix
entries. Matrix truth is written only by the Phase 8 `release-check` probe
pipeline (native backends) and, for web backends, by the still-deferred
browser probe runner. Freshness automation surfaces staleness — it never
overrides the Phase 6 trap rule.
"""
