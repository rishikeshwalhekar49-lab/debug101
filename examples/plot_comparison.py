"""Draw the model comparison from `compare_models.py --json` as one PNG (needs matplotlib).

    python examples/plot_comparison.py comparison.json model_comparison.png

Top: accuracy of every model on the test set nothing was tuned on (the Colab test, when
the data has one; the hand-written test otherwise) and on held-out generated scripts,
with 95% Wilson intervals, so the gap between the two is visible. Bottom: accuracy
against model size (log scale).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from silent_bug_agent.learned.features import FEATURES  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated together
OURS = "Trace classifier (ours, shipped)"
SHORT = {
    "Trace classifier (ours, shipped)": "Ours: trace classifier",
    "Trace classifier with a hidden layer (ours)": "Ours, with a hidden layer",
    "Trace classifier, no hidden layer (ours)": "Ours, no hidden layer",
    "Random forest (300 trees)": "Random forest",
    "Gradient boosting": "Gradient boosting",
    "k-nearest neighbours": "k-nearest neighbours",
    "SVM (RBF kernel)": "SVM (RBF)",
    "Hand-written rules (this project's detectors)": "Hand-written rules",
    "Majority class": "Always the commonest label",
}


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def size_label(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} KB"


def time_label(seconds: float) -> str:
    return f"{seconds * 1e3:.1f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.0f} µs"


# where each size-panel label sits: (x offset, y offset in points, horizontal alignment)
PLACE = {OURS: (10, -11, "left"), "Trace classifier with a hidden layer (ours)": (9, 9, "left"),
         "Trace classifier, no hidden layer (ours)": (10, 11, "left"),
         "SVM (RBF kernel)": (-9, -9, "right"), "k-nearest neighbours": (-10, 0, "right"),
         "Random forest (300 trees)": (9, 9, "left"), "Gradient boosting": (10, 0, "left")}
POINT_NAMES = {OURS: "Ours", "Trace classifier with a hidden layer (ours)": "Ours, hidden layer",
               "Trace classifier, no hidden layer (ours)": "Ours, no hidden layer",
               "SVM (RBF kernel)": "SVM", "k-nearest neighbours": "k-NN", "Random forest (300 trees)": "Random forest",
               "Gradient boosting": "Gradient boosting"}


TEST = "test"  # set in main(): "colab_test" when the data has the Colab test


def accuracy(r: Dict, split: str) -> float:
    """Mean over seeds when the model was retrained, else the single measurement."""
    split = TEST if split == "test" else split
    seeds = r.get("over_seeds")
    key = {"test": "test_accuracy", "generated": "generated_accuracy", "colab_test": "colab_test_accuracy"}[split]
    if seeds and key in seeds:
        return seeds[key]["mean"]
    metric = ((r.get("subsets") or {}).get(split) if split == "colab_test" else r[split]) or {}
    return metric.get("accuracy")  # None: not measured (the rules are not run on the generated scripts)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=10.5)


def main(source: str, out: str) -> int:
    global TEST
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    results: List[Dict] = data["results"]
    counts = dict(data["counts"])
    colab = "colab_test" in counts
    if colab:
        TEST = "colab_test"
        counts["test"] = counts["colab_test"]
        counts["test_healthy"] = counts["colab_test_healthy"]
    test_name = "Colab test" if colab else "Hand-written test"
    order = sorted(results, key=lambda r: accuracy(r, "test"))
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=SURFACE)
    top = fig.add_axes([0.37, 0.52, 0.58, 0.31])
    bottom = fig.add_axes([0.14, 0.15, 0.81, 0.23])

    # ---- top: accuracy on both test sets
    style(top)
    for x in (0, 0.25, 0.5, 0.75, 1.0):
        top.axvline(x, color=GRID, linewidth=0.8, zorder=0)
    for row, r in enumerate(order):
        pairs = [pair for pair in ((0.12, accuracy(r, "test"), counts["test"], ORANGE),
                                   (-0.12, accuracy(r, "generated"), counts["generated"], BLUE)) if pair[1] is not None]
        for offset, value, n, colour in pairs:
            lo, hi = wilson(value, n)
            top.plot([lo, hi], [row + offset, row + offset], color=colour, linewidth=1.5, alpha=0.55, zorder=2,
                     solid_capstyle="round")
            top.scatter([value], [row + offset], s=46, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        if r["model"] == OURS:  # values for our model only: above its orange dot, below its blue one
            for (offset, value, _, _), dy, va in zip(pairs, (4, -3), ("bottom", "top")):
                top.annotate(f"{value:.0%}", (value, row + offset), xytext=(0, dy), textcoords="offset points",
                             ha="center", va=va, fontsize=10, color=INK, fontweight="bold")
    top.set_yticks(range(len(order)))
    top.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in order], fontsize=10.5)
    for label, r in zip(top.get_yticklabels(), order):
        if r["model"].startswith("Trace classifier"):
            label.set_fontweight("bold")
            label.set_color(INK)
    top.set_xlim(0, 1.02)
    top.set_ylim(-0.6, len(order) - 0.4)
    top.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    top.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    top.legend(handles=[Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=7,
                               label=f"{test_name} (n = {counts['test']})"),
                        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7,
                               label=f"Held-out generated scripts (n = {counts['generated']})")],
               loc="lower left", bbox_to_anchor=(-0.58, 1.0), ncol=2, frameon=False, fontsize=10,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)

    # ---- bottom: accuracy against size
    style(bottom)
    sized = [r for r in results if r.get("bytes") and r["model"] != "Majority class"]
    bottom.set_xscale("log")
    # the y-axis fits the main group; a model far below it is drawn on the bottom edge, marked as off the scale
    values = sorted(accuracy(r, "test") for r in sized)
    main_group = [v for v in values if v >= values[-1] - 0.15]
    low = math.floor((min(main_group) - 0.012) * 100) / 100
    high = math.ceil((max(main_group) + 0.012) * 100) / 100
    for r in sized:
        ours = "(ours" in r["model"]
        colour = BLUE if ours else MUTED
        x, y = r["bytes"], accuracy(r, "test")
        off_scale = y < low
        shown_y = low + 0.004 if off_scale else y
        bottom.scatter([x], [shown_y], s=64 if ours else 46, color=colour, edgecolors=SURFACE, linewidths=1.5,
                       zorder=3, marker="v" if off_scale else "o", clip_on=False)
        text = f"{POINT_NAMES.get(r['model'], r['model'])} · {size_label(x)}"
        if off_scale:
            text += f" · {y:.0%} (off the axis)"
        dx, dy, ha = PLACE.get(r["model"], (9, 9, "left"))
        bottom.annotate(text, (x, shown_y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=9.2,
                        color=INK if r["model"] == OURS else INK_2,
                        fontweight="bold" if r["model"] == OURS else "normal")
    bottom.set_xlim(1.5 * 1024, 300 * 1024 * 1024)
    bottom.set_ylim(low, high)
    ticks = [t / 100 for t in range(math.ceil(low * 100), math.floor(high * 100) + 1) if t % 2 == 0]
    bottom.set_xticks([1024, 10 * 1024, 100 * 1024, 1024 ** 2, 10 * 1024 ** 2, 100 * 1024 ** 2])
    bottom.set_xticklabels(["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"])
    bottom.minorticks_off()
    bottom.set_yticks(ticks)
    bottom.set_yticklabels([f"{t:.0%}" for t in ticks])
    for t in ticks:
        bottom.axhline(t, color=GRID, linewidth=0.8, zorder=0)
    bottom.set_xlabel("Model size (log scale)", color=INK_2, fontsize=10.5, labelpad=6)
    bottom.set_ylabel(f"Accuracy, {test_name.lower()}\n(n = {counts['test']})", color=INK_2, fontsize=10.5)

    # ---- titles and notes (text in ink, never in series colours)
    ours = next(r for r in results if r["model"] == OURS)
    fig.text(0.05, 0.955, f"A {ours['bytes'] / 1024:.0f} KB model that names the bug in a training run", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.93, "Nine silent PyTorch bugs or “clean”, from 150 steps of training statistics and its code"
             if colab else "Nine silent PyTorch bugs or “clean”, from 60 steps of training statistics",
             fontsize=11, color=INK_2)
    fig.text(0.05, 0.875, "Accuracy by model", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.05, 0.415, "Accuracy against model size", fontsize=12, fontweight="bold", color=INK)
    if colab:
        note = (f"Colab test: {counts['test']} runs of 6 scripts from Google Colab notebooks on Colab's sample data,"
                f" never used to choose anything ({counts['test_healthy']} healthy).\nSettings chosen on {counts['dev']}"
                f" tuning runs. Every learned model gets the same {len(FEATURES)} features (trace and code); mean of 5"
                " seeds.\nLines: 95% Wilson intervals. Rules have no model file. LLMs (gpt-oss) not tested yet."
                " CPU (Apple M2), 25 Nov 2026.")
    else:
        note = (f"Hand-written test: {counts['test']} runs never used to choose anything ({counts['test_healthy']}"
                f" healthy). Settings chosen on {counts['dev']} other hand-written runs.\nEvery learned model gets the"
                f" same {len(FEATURES)} features; mean of 5 seeds. Lines: 95% Wilson intervals. Rules have no model"
                " file.\nLLMs (gpt-oss) not tested yet. CPU (Apple M2), 24 Nov 2026.")
    fig.text(0.05, 0.018, note, fontsize=7.6, color=MUTED, linespacing=1.4)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
