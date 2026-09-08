"""Manual-verify register (`MAINTENANCE.md`) parsing.

The register lists what cannot be CI'd — on-device card benchmarks, matrix
entries that `release-check` reports as needing a device attached or as
unprobeable — each with a re-verify interval. `reminders.yml` runs
`edge-compat freshness overdue` monthly and opens or bumps one reminder
issue per overdue item.

Register rows are a markdown table with exactly these columns:
`| id | item | interval_days | status | last_verified | notes |`
`status` is `active` or `dormant` (dormant rows are registered obligations
that have no real data behind them yet; they are never overdue).
`last_verified` is `YYYY-MM-DD` or `never`.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

_EXPECTED_COLUMNS = ["id", "item", "interval_days", "status", "last_verified", "notes"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RegisterError(ValueError):
    """MAINTENANCE.md is missing or its register table is malformed."""


@dataclass(frozen=True)
class RegisterItem:
    id: str
    item: str
    interval_days: int
    status: str
    last_verified: str  # YYYY-MM-DD or 'never'
    notes: str

    def due_by(self) -> datetime.date | None:
        """The date on or after which this item is overdue; None for dormant
        items. An active item never verified is due immediately."""
        if self.status != "active":
            return None
        if self.last_verified == "never":
            return datetime.date.min
        verified = datetime.date.fromisoformat(self.last_verified)
        return verified + datetime.timedelta(days=self.interval_days)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_register(path: Path) -> list[RegisterItem]:
    """Parse the first register table in MAINTENANCE.md. Strict: a malformed
    row is an error — a silently skipped obligation is exactly the decay this
    file exists to prevent."""
    if not path.is_file():
        raise RegisterError(f"register not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if _split_row(line) == _EXPECTED_COLUMNS), None
    )
    if header_index is None or header_index + 1 >= len(lines):
        raise RegisterError(
            f"{path}: no register table with columns {' | '.join(_EXPECTED_COLUMNS)}"
        )
    items: list[RegisterItem] = []
    errors: list[str] = []
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _split_row(line)
        if len(cells) != len(_EXPECTED_COLUMNS):
            errors.append(f"row has {len(cells)} cells, expected {len(_EXPECTED_COLUMNS)}: {line}")
            continue
        row = dict(zip(_EXPECTED_COLUMNS, cells, strict=True))
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", row["id"]):
            errors.append(f"{row['id']!r}: id must be kebab-case")
        if not row["interval_days"].isdigit() or int(row["interval_days"]) < 1:
            errors.append(f"{row['id']!r}: interval_days must be a positive integer")
            continue
        if row["status"] not in ("active", "dormant"):
            errors.append(f"{row['id']!r}: status must be 'active' or 'dormant'")
            continue
        if row["last_verified"] != "never" and not _DATE_RE.match(row["last_verified"]):
            errors.append(f"{row['id']!r}: last_verified must be YYYY-MM-DD or 'never'")
            continue
        items.append(
            RegisterItem(
                id=row["id"],
                item=row["item"],
                interval_days=int(row["interval_days"]),
                status=row["status"],
                last_verified=row["last_verified"],
                notes=row["notes"],
            )
        )
    duplicates = {i.id for i in items if sum(1 for j in items if j.id == i.id) > 1}
    errors.extend(f"duplicate register id {d!r}" for d in sorted(duplicates))
    if errors:
        raise RegisterError(f"{path}: invalid register:\n" + "\n".join(f"  {e}" for e in errors))
    return items


def overdue_items(items: list[RegisterItem], as_of: datetime.date) -> list[RegisterItem]:
    return [
        item for item in items if (due := item.due_by()) is not None and due <= as_of
    ]
