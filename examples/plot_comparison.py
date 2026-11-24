"""Draw the model comparison from `compare_models.py --json` as one PNG (needs matplotlib).

    python examples/plot_comparison.py comparison.json model_comparison.png

Top: accuracy of every model on the two test sets, with 95% Wilson intervals, so the
gap between generated and hand-written scripts is visible. Bottom: accuracy against
model size (log scale), with prediction time in each label.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated together
OURS = "Trace classifier (ours, as shipped)"
SHORT = {
    "Trace classifier (ours, as shipped)": "Ours: trace classifier",
    "Softmax regression (ours, no hidden layer)": "Ours, no hidden layer",
    "Random forest (300 trees)": "Random forest",
    "Gradient boosting": "Gradient boosting",
    "k-nearest neighbours (k=5)": "k-nearest neighbours",
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


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=10.5)


def main(source: str, out: str) -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    results: List[Dict] = data["results"]
    order = sorted(results, key=lambda r: r["templates"]["accuracy"])
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=SURFACE)
    top = fig.add_axes([0.37, 0.52, 0.58, 0.31])
    bottom = fig.add_axes([0.14, 0.15, 0.81, 0.23])

    # ---- top: accuracy on both test sets
    style(top)
    for x in (0, 0.25, 0.5, 0.75, 1.0):
        top.axvline(x, color=GRID, linewidth=0.8, zorder=0)
    for row, r in enumerate(order):
        a, b = r["templates"], r["benchmarks"]
        for offset, metric, colour in ((0.12, a, BLUE), (-0.12, b, ORANGE)):
            lo, hi = wilson(metric["accuracy"], metric["n"])
            top.plot([lo, hi], [row + offset, row + offset], color=colour, linewidth=1.5, alpha=0.55, zorder=2,
                     solid_capstyle="round")
            top.scatter([metric["accuracy"]], [row + offset], s=46, color=colour, edgecolors=SURFACE,
                        linewidths=1.5, zorder=3)
        if r["model"] == OURS:  # values for our model only: above its blue dot, below its orange one
            for offset, metric, dy, va in ((0.12, a, 4, "bottom"), (-0.12, b, -3, "top")):
                top.annotate(f"{metric['accuracy']:.0%}", (metric["accuracy"], row + offset), xytext=(0, dy),
                             textcoords="offset points", ha="center", va=va, fontsize=10, color=INK,
                             fontweight="bold")
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
    n_templates = order[0]["templates"]["n"]
    n_bench = order[0]["benchmarks"]["n"]
    top.legend(handles=[Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7,
                               label=f"Held-out generated scripts (n = {n_templates})"),
                        Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=7,
                               label=f"Hand-written benchmarks (n = {n_bench})")],
               loc="lower left", bbox_to_anchor=(-0.58, 1.0), ncol=2, frameon=False, fontsize=10,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)

    # ---- bottom: accuracy against size
    style(bottom)
    sized = [r for r in results if r.get("bytes") and r["model"] != "Majority class"]
    bottom.set_xscale("log")
    for y in (0.85, 0.9, 0.95):
        bottom.axhline(y, color=GRID, linewidth=0.8, zorder=0)
    # hand-placed labels: (x offset, y offset in points, horizontal alignment)
    placement = {"Trace classifier (ours, as shipped)": (9, 8, "left"),
                 "Softmax regression (ours, no hidden layer)": (9, -9, "left"),
                 "Random forest (300 trees)": (9, 8, "left"),
                 "Gradient boosting": (9, -10, "left"),
                 "SVM (RBF kernel)": (-9, 8, "right"),
                 "k-nearest neighbours (k=5)": (9, -9, "left")}
    for r in sized:
        ours = r["model"].startswith("Trace classifier") or "(ours" in r["model"]
        colour = BLUE if ours else MUTED
        x, y = r["bytes"], r["templates"]["accuracy"]
        bottom.scatter([x], [y], s=64 if ours else 46, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        name = {OURS: "Ours"}.get(r["model"], SHORT.get(r["model"], r["model"]))
        text = f"{name} · {size_label(x)} · {time_label(r['seconds_per_prediction'])}"
        dx, dy, ha = placement.get(r["model"], (9, 0, "left"))
        bottom.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=9.2, color=INK if r["model"] == OURS else INK_2,
                        fontweight="bold" if r["model"] == OURS else "normal")
    bottom.set_xlim(1.5 * 1024, 300 * 1024 * 1024)
    bottom.set_ylim(0.83, 0.97)
    bottom.set_xticks([1024, 10 * 1024, 100 * 1024, 1024 ** 2, 10 * 1024 ** 2, 100 * 1024 ** 2])
    bottom.set_xticklabels(["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"])
    bottom.minorticks_off()
    bottom.set_yticks([0.85, 0.9, 0.95])
    bottom.set_yticklabels(["85%", "90%", "95%"])
    bottom.set_xlabel("Model size (log scale)", color=INK_2, fontsize=10.5, labelpad=6)
    bottom.set_ylabel("Accuracy, held-out\ngenerated scripts", color=INK_2, fontsize=10.5)

    # ---- titles and notes (text in ink, never in series colours)
    fig.text(0.05, 0.955, "A 12 KB model that names the bug in a training run", fontsize=16, fontweight="bold",
             color=INK)
    fig.text(0.05, 0.93, "Nine silent PyTorch bugs or “clean”, from 60 steps of training statistics",
             fontsize=11, color=INK_2)
    fig.text(0.05, 0.875, "Accuracy by model", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.05, 0.415, "Accuracy against size, with time per prediction", fontsize=12, fontweight="bold",
             color=INK)
    fig.text(0.05, 0.018,
             "Every learned model gets the same 24 features. Generated scripts: 5-fold cross-validation, whole "
             "templates held out.\nBenchmarks: blind copies of the project's hand-written scripts (9 buggy, 17 "
             "healthy). Lines: 95% Wilson intervals.\nHand-written rules need a 150-step run per script and have "
             "no size. LLMs (gpt-oss) not tested yet. CPU (Apple M2), 24 Nov 2026.",
             fontsize=7.6, color=MUTED, linespacing=1.4)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
