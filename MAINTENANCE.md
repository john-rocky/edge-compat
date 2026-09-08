# MAINTENANCE — manual-verify register (Phase 11)

What cannot be CI'd, each with a re-verify interval. `reminders.yml` runs
`edge-compat freshness overdue --register MAINTENANCE.md` monthly and opens
or bumps one reminder issue per overdue item — scheduled reminders instead of
silent decay.

Row contract (parsed by `litert_compat.freshness.maintenance`, strict):
`id` kebab-case and unique; `interval_days` a positive integer; `status`
`active` (reminded when overdue) or `dormant` (a registered obligation with no
real data behind it yet — never overdue; flip to `active` when the data
lands); `last_verified` `YYYY-MM-DD` or `never`. Update `last_verified` by
hand after each manual re-verification — this register is owner-maintained;
only the reminder issues are automated.

| id | item | interval_days | status | last_verified | notes |
|---|---|---|---|---|---|
| card-bench-ondevice | On-device card benchmarks (physical devices; cannot be CI'd) | 90 | active | 2026-08-25 | activated 2026-08-26 (owner-approved): S4 galaxy-s26 GPU rows attached to 22 cards on 2026-08-25 |
| matrix-adb-probes | Matrix entries `release-check` lists as needing an attached Android device (`gpu_mldrift_adb` runner) | 60 | dormant | never | activates with the first real matrix snapshots + the owner-side adb helper (Phase 8 hand-off 3) |
| matrix-unprobeable | Matrix entries `release-check` reports as unprobeable (weighted ops, option-constrained signatures) | 90 | dormant | never | re-verify by full-model measurement until the probe registry extensions land (Deferred) |
| webnn-experimental | `webnn` sweep lane review — API stability and whether results go public (Phase 5 hand-off 3) | 90 | dormant | never | harness measures it behind `--experimental` today |
| device-run-remeasure | Device-run snapshots re-measured on new litert / litert-lm releases (physical devices; cannot be CI'd) | 90 | active | 2026-08-25 | activated 2026-08-26 (owner-approved): snapshots span 0.13.1..0.16.1 + selfbuilt, last measured 2026-08-25; autobump keeps the `litertlm` axis current so staleness is visible |
