"""Draw the comparison as a ranked leaderboard table, terminal style (needs matplotlib).

    python examples/compare_models.py --data traces.jsonl --seeds 5 --reuse-rules comparison.json --json lb.json
    python examples/plot_leaderboard.py lb.json leaderboard.png

Every number is from `compare_models.py`. The two LLM rows give only OpenAI's published
parameter counts: their accuracy has not been measured, and the table says so.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKGROUND, INK, DIM, RULE = "#0f0f0f", "#e9e9e6", "#9a9a95", "#d6d6d0"
FONT = ["Menlo", "DejaVu Sans Mono", "Courier New"]
SIZE = 10.5  # points
CHAR = 0.602 * SIZE  # advance width of one monospace character, points
LINE = 1.45 * SIZE  # line height, points
OUR_BYTES, OUR_PARAMS = 11.7 * 1024, 1130
LLMS = [("gpt-oss-120b (LLM, via Groq)", 117e9), ("gpt-oss-20b (LLM, via Groq)", 21e9)]  # OpenAI's published counts

NAMES = {
    "Trace classifier (ours, as shipped)": "Trace classifier (ours)",
    "Softmax regression (ours, no hidden layer)": "Trace classifier, no hidden layer (ours)",
    "Random forest (300 trees)": "Random forest (300 trees)",
    "Gradient boosting": "Gradient boosting",
    "SVM (RBF kernel)": "SVM (RBF kernel)",
    "k-nearest neighbours (k=5)": "k-nearest neighbours (k=5)",
    "Hand-written rules (this project's detectors)": "Hand-written rules (the project's detectors)",
    "Majority class": "Always the commonest label",
}
TRAINED = {"Hand-written rules (this project's detectors)": "no: written by hand",
           "Majority class": "label counts only"}
COLUMNS = [("Rank", 4), ("Model", 26), ("Size", 11), ("x ours", 8), ("Trained on this task?", 13),
           ("New scripts, 701 runs (%)", 13), ("Hand-written, 26 runs (%)", 13), ("Healthy flagged", 9),
           ("CPU time / prediction", 11), ("Number from", 15)]


def size_text(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def ratio_text(r: float) -> str:
    if r >= 1e6:
        return f"{r / 1e6:,.1f}M×"
    return f"{r:,.0f}×" if r >= 10 else f"{r:.1f}×"


def time_text(s: float) -> str:
    if s >= 1:
        return f"{s:.1f} s *"
    return f"{s * 1e3:.1f} ms" if s >= 1e-3 else f"{s * 1e6:.0f} µs"


def pct(spread: Dict[str, float] | None, value: float) -> str:
    if spread is None:
        return f"{value * 100:.1f}"
    return f"{spread['mean'] * 100:.1f} ± {spread['std'] * 100:.1f}"


def rows_from(results: List[Dict[str, Any]], n_healthy: int) -> List[List[str]]:
    measured = sorted(results, key=lambda r: -(r.get("over_seeds", {}).get("templates_accuracy", {}).get("mean")
                                              or r["templates"]["accuracy"]))
    rows = []
    for rank, r in enumerate(measured, start=1):
        # a constant guess has no randomness, so no spread is shown for it
        seeds = r.get("over_seeds") if r["model"] != "Majority class" else None
        healthy = r["benchmarks"]["false_alarms_on_healthy"]
        size = size_text(r["bytes"]) if r.get("bytes") and r["model"] != "Majority class" else "-"
        ratio = ratio_text(r["bytes"] / OUR_BYTES) if size != "-" else "-"
        rows.append([str(rank), NAMES.get(r["model"], r["model"]), size, ratio,
                     TRAINED.get(r["model"], "yes"),
                     pct(seeds and seeds["templates_accuracy"], r["templates"]["accuracy"]),
                     pct(seeds and seeds["benchmarks_accuracy"], r["benchmarks"]["accuracy"]),
                     f"{round((seeds['false_alarms_on_healthy']['mean'] if seeds else healthy) * n_healthy)} of {n_healthy}",
                     time_text(r["seconds_per_prediction"]) if r["model"] != "Majority class" else "-",
                     f"measured by us, {seeds['templates_accuracy']['runs']} seeds" if seeds else "measured by us"])
    for name, params in LLMS:
        rows.append(["-", name, f"{params / 1e9:.0f}B params", ratio_text(params / OUR_PARAMS), "no",
                     "not measured yet", "not measured yet", "-", "-", "needs an API key"])
    return rows


def main(source: str, out: str) -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    rows = rows_from(data["results"], sum(label == "clean" for label in data["benchmark_labels"]))
    widths = [w for _, w in COLUMNS]
    pad = 1  # characters of padding on each side of a cell
    header = [textwrap.wrap(title, w) for title, w in COLUMNS]
    cells = [[textwrap.wrap(text, w) or [""] for text, w in zip(row, widths)] for row in rows]
    heights = [max(len(c) for c in header)] + [max(len(c) for c in row) for row in cells]

    total_chars = sum(w + 2 * pad for w in widths)
    title_lines, note_lines = 2, 6
    row_lines = sum(h + 1 for h in heights)
    width_pt = (total_chars + 4) * CHAR
    height_pt = (title_lines + row_lines + note_lines + 2) * LINE
    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), dpi=150, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_pt)
    ax.set_ylim(height_pt, 0)
    ax.axis("off")
    ax.set_facecolor(BACKGROUND)

    def text(x: float, y: float, s: str, colour: str = INK, bold: bool = False) -> None:
        ax.text(x, y, s, fontsize=SIZE, family=FONT, color=colour, va="top", ha="left",
                fontweight="bold" if bold else "normal")

    left = 2 * CHAR
    top = LINE * 0.8
    text(left, top, "1. Naming the bug in a PyTorch training run "
                    "(9 silent bugs or \"clean\", top-1 accuracy)", bold=True)
    y = top + title_lines * LINE
    xs = [left]
    for w in widths:
        xs.append(xs[-1] + (w + 2 * pad) * CHAR)
    table_top = y
    ys = [y]
    for h in heights:
        ys.append(ys[-1] + (h + 1) * LINE)
    for x in xs:
        ax.plot([x, x], [table_top, ys[-1]], color=RULE, linewidth=0.9)
    for yy in ys:
        ax.plot([xs[0], xs[-1]], [yy, yy], color=RULE, linewidth=0.9)
    for column, lines in enumerate(header):
        for i, line in enumerate(lines):
            text(xs[column] + pad * CHAR, ys[0] + (0.5 + i) * LINE, line)
    for r, (row, raw) in enumerate(zip(cells, rows)):
        ours = "(ours)" in raw[1]
        pending = raw[5] == "not measured yet"
        for column, lines in enumerate(row):
            for i, line in enumerate(lines):
                text(xs[column] + pad * CHAR, ys[r + 1] + (0.5 + i) * LINE, line,
                     colour=DIM if pending else INK, bold=ours)

    notes = [
        "New scripts: 5-fold cross-validation over 80 generated training scripts, whole templates held out;"
        " mean ± std over 5 seeds.",
        "Hand-written: blind copies of the project's own benchmark scripts (9 buggy, 17 healthy), never trained on;"
        " 21 of 26 right is 62-92% at 95%.",
        "x ours: file size (ours 11.7 KB). For the LLMs, parameter count (ours 1,130; OpenAI's published counts).",
        "* the rules' time includes the 150-step training run they need. The learned models also need one 60-step"
        " traced run (about 4 s).",
        "Learned models all get the same 24 trace features. Apple M2 CPU, 24 Nov 2026."
        " Code: github.com/rishikeshwalhekar49-lab/debug101",
    ]
    for i, note in enumerate(notes):
        text(left, ys[-1] + (0.9 + i) * LINE, note, colour=DIM)
    fig.savefig(out, dpi=150, facecolor=BACKGROUND)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
