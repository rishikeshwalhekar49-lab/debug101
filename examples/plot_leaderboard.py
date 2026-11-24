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
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from silent_bug_agent.learned.features import FEATURES  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKGROUND, INK, DIM, RULE = "#0f0f0f", "#e9e9e6", "#9a9a95", "#d6d6d0"
FONT = ["Menlo", "DejaVu Sans Mono", "Courier New"]
SIZE = 10.5  # points
CHAR = 0.602 * SIZE  # advance width of one monospace character, points
LINE = 1.45 * SIZE  # line height, points
LLMS = [("gpt-oss-120b (LLM, via Groq)", 117e9), ("gpt-oss-20b (LLM, via Groq)", 21e9)]  # OpenAI's published counts

NAMES = {
    "Trace classifier (ours, shipped)": "Trace classifier (ours, shipped)",
    "Trace classifier with a hidden layer (ours)": "Trace classifier, hidden layer (ours)",
    "Random forest (300 trees)": "Random forest (300 trees)",
    "Gradient boosting": "Gradient boosting",
    "SVM (RBF kernel)": "SVM (RBF kernel)",
    "k-nearest neighbours": "k-nearest neighbours",
    "Hand-written rules (this project's detectors)": "Hand-written rules (the project's detectors)",
    "Majority class": "Always the commonest label",
}
TRAINED = {"Hand-written rules (this project's detectors)": "no: written by hand",
           "Majority class": "label counts only"}


def columns(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    return [("Rank", 4), ("Model", 26), ("Size", 11), ("x ours", 8), ("Trained on this task?", 13),
            (f"Hand-written test, {counts['test']} runs (%)", 14), (f"Generated, {counts['generated']} runs (%)", 13),
            (f"Healthy flagged (of {counts['test_healthy']})", 10), ("CPU time / prediction", 11),
            ("Number from", 15)]


def size_text(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def ratio_text(r: float) -> str:
    if r >= 1e6:
        return f"{r / 1e6:,.1f}M×"
    return f"{r:,.0f}×" if r >= 10 else f"{r:.1f}×"


def time_text(s: Optional[float]) -> str:
    if s is None:
        return "-"
    if s >= 1:
        return f"{s:.1f} s *"
    return f"{s * 1e3:.1f} ms" if s >= 1e-3 else f"{s * 1e6:.0f} µs"


def score(r: Dict[str, Any], split: str) -> Tuple[Optional[float], Optional[float]]:
    seeds = r.get("over_seeds") if r["model"] != "Majority class" else None
    key = {"test": "test_accuracy", "generated": "generated_accuracy"}[split]
    if seeds:
        return seeds[key]["mean"], seeds[key]["std"]
    metric = r.get(split) or {}
    return (metric.get("accuracy"), None) if metric.get("n") else (None, None)


def pct(value: Tuple[Optional[float], Optional[float]]) -> str:
    mean, std = value
    if mean is None:
        return "-"
    return f"{mean * 100:.1f}" if std is None else f"{mean * 100:.1f} ± {std * 100:.1f}"


def rows_from(results: List[Dict[str, Any]], counts: Dict[str, int]) -> List[List[str]]:
    ours_bytes = next(r["bytes"] for r in results if r["model"] == "Trace classifier (ours, shipped)")
    ours_params = next(r["parameters"] for r in results if r["model"] == "Trace classifier (ours, shipped)")
    ranked = sorted(results, key=lambda r: -(score(r, "test")[0] or 0.0))
    rows = []
    for rank, r in enumerate(ranked, start=1):
        seeds = r.get("over_seeds") if r["model"] != "Majority class" else None
        alarms = seeds["test_false_alarms"]["mean"] if seeds else r["test"]["false_alarms_on_healthy"]
        size = size_text(r["bytes"]) if r.get("bytes") else "-"
        rows.append([str(rank), NAMES.get(r["model"], r["model"]), size,
                     ratio_text(r["bytes"] / ours_bytes) if r.get("bytes") else "-",
                     TRAINED.get(r["model"], "yes"), pct(score(r, "test")), pct(score(r, "generated")),
                     f"{round(alarms * counts['test_healthy'])}",
                     time_text(r["seconds_per_prediction"]) if r["model"] != "Majority class" else "-",
                     f"measured by us, {seeds['test_accuracy']['runs']} seeds" if seeds else "measured by us"])
    for name, params in LLMS:
        rows.append(["-", name, f"{params / 1e9:.0f}B params", ratio_text(params / ours_params), "no",
                     "not measured yet", "not measured yet", "-", "-", "needs an API key"])
    return rows


def draw_table(title: str, cols: List[Tuple[str, int]], rows: List[List[str]], notes: List[str], out: str,
               bold: Any = lambda row: False, dim: Any = lambda row: False) -> None:
    """A terminal-style table: monospace text, light rules on a dark background."""
    widths = [w for _, w in cols]
    pad = 1  # characters of padding on each side of a cell
    header = [textwrap.wrap(name, w) for name, w in cols]
    cells = [[textwrap.wrap(text, w) or [""] for text, w in zip(row, widths)] for row in rows]
    heights = [max(len(c) for c in header)] + [max(len(c) for c in row) for row in cells]
    width_pt = (sum(w + 2 * pad for w in widths) + 4) * CHAR
    note_lines = sum(len(textwrap.wrap(note, sum(w + 2 * pad for w in widths))) for note in notes)
    height_pt = (2 + sum(h + 1 for h in heights) + note_lines + 3) * LINE
    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), dpi=150, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_pt)
    ax.set_ylim(height_pt, 0)
    ax.axis("off")

    def text(x: float, y: float, s: str, colour: str = INK, heavy: bool = False) -> None:
        ax.text(x, y, s, fontsize=SIZE, family=FONT, color=colour, va="top", ha="left",
                fontweight="bold" if heavy else "normal")

    left, top = 2 * CHAR, LINE * 0.8
    text(left, top, title, heavy=True)
    xs = [left]
    for w in widths:
        xs.append(xs[-1] + (w + 2 * pad) * CHAR)
    ys = [top + 2 * LINE]
    for h in heights:
        ys.append(ys[-1] + (h + 1) * LINE)
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], color=RULE, linewidth=0.9)
    for yy in ys:
        ax.plot([xs[0], xs[-1]], [yy, yy], color=RULE, linewidth=0.9)
    for column, lines in enumerate(header):
        for i, line in enumerate(lines):
            text(xs[column] + pad * CHAR, ys[0] + (0.5 + i) * LINE, line)
    for r, (row, raw) in enumerate(zip(cells, rows)):
        for column, lines in enumerate(row):
            for i, line in enumerate(lines):
                text(xs[column] + pad * CHAR, ys[r + 1] + (0.5 + i) * LINE, line,
                     colour=DIM if dim(raw) else INK, heavy=bold(raw))
    wrapped = [line for note in notes for line in textwrap.wrap(note, sum(w + 2 * pad for w in widths))]
    for i, line in enumerate(wrapped):
        text(left, ys[-1] + (0.9 + i) * LINE, line, colour=DIM)
    fig.savefig(out, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def main(source: str, out: str) -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    counts = data["counts"]
    ours = next(r for r in data["results"] if r["model"] == "Trace classifier (ours, shipped)")
    benchmark_runs = sum(group.startswith("benchmark") for group in data["test_groups"])
    notes = [
        f"Hand-written test: {counts['test']} runs never used to choose anything: 6 scripts written separately from the"
        f" generator ({counts['test'] - benchmark_runs} versions) and blind copies of the project's benchmarks"
        f" ({benchmark_runs} versions: the buggy ones whose bug the classifier knows, and all the healthy twins).",
        f"Generated: 5-fold cross-validation over {counts['generated']} runs from 80 generated scripts, whole scripts"
        f" held out. Each model family chose its settings on {counts['dev']} other hand-written runs.",
        f"Mean ± std over 5 seeds. x ours: file size (ours {size_text(ours['bytes'])}); for the LLMs, parameter count"
        f" (ours {ours['parameters']:,}; OpenAI's published counts).",
        "* the rules' time includes the 150-step training run they need. The learned models also need one 60-step"
        " traced run (about 4 s).",
        f"Learned models all get the same {len(FEATURES)} trace features. Apple M2 CPU, 24 Nov 2026."
        " Code: github.com/rishikeshwalhekar49-lab/debug101",
    ]
    draw_table("1. Naming the bug in a PyTorch training run (9 silent bugs or \"clean\", top-1 accuracy, "
               "ranked on the hand-written test)", columns(counts), rows_from(data["results"], counts), notes, out,
               bold=lambda row: "(ours" in row[1], dim=lambda row: row[5] == "not measured yet")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
