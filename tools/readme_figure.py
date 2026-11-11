#!/usr/bin/env python3
"""Draw the picture at the top of README.md from cards/index.json.

    python3 tools/readme_figure.py

Writes docs/images/model-status-light.svg and -dark.svg, and prints the sentence the alt text uses.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "cards", "index.json")
OUT = os.path.join(ROOT, "docs", "images")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "pass": "#2a78d6", "other": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "pass": "#3987e5", "other": "#d95926"},
}
# (device or None for the browser, accelerator, label in the figure)
TARGETS = [
    (None, "webgpu_mldrift", "Browser, WebGPU"),
    (None, "wasm_xnnpack", "Browser, WASM"),
    ("galaxy-s26", "gpu_mldrift", "Galaxy S26, GPU"),
    ("galaxy-s26", "npu_qnn", "Galaxy S26, NPU"),
    ("raspberry-pi-5", "cpu_xnnpack", "Raspberry Pi 5, CPU"),
]
OUTCOME = {"pass": "pass", "fallback": "other", "output_mismatch": "other", "load_failed": "other", "run_failed": "other"}
COLUMNS = [("pass", "Loaded and ran, no fallback or mismatch recorded"), ("other", "Fell back, gave wrong output, or failed to load or run")]
HEADS = ["Ran", "Other"]
TITLE = "Which models ran where"
WIDTH, LABELS, BARS, ROW, BAR, COLUMN = 860, 170, 520, 34, 18, 76


def counts():
    """Return ([{label, pass, other}], newest date) read from the card index."""
    if not os.path.exists(SOURCE):
        sys.exit(f"readme_figure: {SOURCE} not found")
    with open(SOURCE, encoding="utf-8") as f:
        models = [m for m in json.load(f)["models"] if not m["id"].startswith("example-")]
    rows, newest = [], ""
    for device, accelerator, label in TARGETS:
        tally = {"pass": 0, "other": 0}
        for m in models:
            if device is None:
                browser = m.get("browser") or {}
                status = browser.get("statuses", {}).get(accelerator)
                if status is not None:
                    newest = max(newest, browser["sweep_source"].split("/")[3])
            else:
                record = next((r for r in (m.get("device") or {}).get("records", [])
                               if r["device"] == device and r["label"] == accelerator), None)
                status = record["status"] if record else None
                if record:
                    newest = max(newest, record["date"])
            if status is not None:
                tally[OUTCOME[status]] += 1
        if not sum(tally.values()):
            sys.exit(f"readme_figure: no records for {label} in cards/index.json")
        rows.append({"label": label, **tally})
    return rows, newest


def text(x, y, body, size, fill, weight="400", anchor="start"):
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{body}</text>')


def segment(x, y, width, fill, left, right):
    """A bar segment, rounded only at the bar's two ends."""
    r = 4
    a = r if left else 0
    b = r if right else 0
    return (f'<path fill="{fill}" d="M{x + a:g} {y:g}H{x + width - b:g}'
            f'{f"Q{x + width:g} {y:g} {x + width:g} {y + b:g}" if b else ""}V{y + BAR - b:g}'
            f'{f"Q{x + width:g} {y + BAR:g} {x + width - b:g} {y + BAR:g}" if b else ""}H{x + a:g}'
            f'{f"Q{x:g} {y + BAR:g} {x:g} {y + BAR - a:g}" if a else ""}V{y + a:g}'
            f'{f"Q{x:g} {y:g} {x + a:g} {y:g}" if a else ""}Z"/>')


def draw(rows, newest, theme, desc):
    c = THEMES[theme]
    parts = [text(0, 28, TITLE, 20, c["text"], "600")]
    parts.append(text(0, 52, f"Number of .tflite models per target. Last updated {newest}. "
                             "The browser rows ran in headless desktop Chromium.", 14, c["muted"]))
    parts.append(text(0, 72, "Not every model ran on every target, so the rows have different totals.", 14, c["muted"]))
    y = 108
    first = LABELS + BARS + 22
    for i, (slot, _) in enumerate(COLUMNS):
        right = first + (i + 1) * COLUMN - 26
        parts.append(text(right, y, HEADS[i], 13, c["muted"], "600", "end"))
    unit = BARS / max(r["pass"] + r["other"] for r in rows)
    for n, r in enumerate(rows):
        top = y + 12 + n * ROW
        parts.append(text(0, top + 14, r["label"], 14, c["text"]))
        shown = [slot for slot, _ in COLUMNS if r[slot]]
        x = LABELS
        for slot in shown:
            w = r[slot] * unit
            parts.append(segment(x, top, max(w - 2, 2), c[slot], slot == shown[0], slot == shown[-1]))
            x += w
        for i, (slot, _) in enumerate(COLUMNS):
            parts.append(text(first + (i + 1) * COLUMN - 26, top + 14, f"{r[slot]:,}", 14, c["text"], "600", "end"))
    y += 12 + len(rows) * ROW + 12
    x = 0
    for slot, label in COLUMNS:
        parts.append(f'<rect x="{x}" y="{y - 11}" width="12" height="12" rx="3" fill="{c[slot]}"/>')
        parts.append(text(x + 18, y, label, 14, c["muted"]))
        x += 18 + 7 * len(label) + 28
    height = y + 10
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    rows, newest = counts()
    desc = TITLE + ". " + " ".join(
        f"{r['label']}: {r['pass']} loaded and ran with no fallback or mismatch recorded, {r['other']} fell back, gave wrong output, or failed to load or run."
        for r in rows)
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"model-status-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(draw(rows, newest, theme, desc))
    print(desc)


if __name__ == "__main__":
    main()
