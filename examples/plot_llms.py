"""Draw the trace classifier against public LLMs, from `compare_llms.py --json`.

    python examples/plot_llms.py docs/images/llm_comparison.json docs/images/llm_comparison.png

Left: accuracy on the Colab test for our model and for each LLM, given the script only
and given the script plus the run's statistics, with 95% Wilson intervals. Right: the
same accuracy against the number of parameters (log scale). Only models that answered
every run are drawn; the others are listed as not measured.
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # the reference palette's first three slots
KINDS = {"code": ("the script only", BLUE), "code+stats": ("the script and the run's statistics", ORANGE)}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def params_text(n: float) -> str:
    return f"{n / 1e9:.0f}B" if n >= 1e9 else f"{n:,.0f}"


def main(source: str, out: str, set_name: str = "Colab test") -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    results = [r for r in data["results"] if r["set"] == set_name]
    ours = next(r for r in results if r["kind"] == "trained")
    llms = [r for r in results if r["kind"] != "trained"]
    measured = [r for r in llms if r.get("complete")]
    missing = sorted({r["model"] for r in llms if not r.get("complete")})
    names = sorted({r["model"] for r in measured},
                   key=lambda m: -max(r["accuracy"] for r in measured if r["model"] == m))

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(length=0, colors=MUTED, labelcolor=INK)

    # ---- left: accuracy, our model first, then each LLM with its two inputs
    rows: List[Tuple[str, str, Dict[str, Any]]] = [("Ours: trace classifier v4", "trained", ours)]
    for name in names:
        for kind in KINDS:
            entry = next((r for r in measured if r["model"] == name and r["kind"] == kind), None)
            if entry:
                rows.append((name, kind, entry))
    y = 0.0
    positions: Dict[str, List[float]] = {}
    previous = None
    for name, kind, entry in rows:
        if previous is not None and name != previous:
            y += 0.45  # a gap between models
        colour = AQUA if kind == "trained" else KINDS[kind][1]
        share = entry["correct"] / entry["n"]
        low, high = wilson(entry["correct"], entry["n"])
        left.barh(y, share * 100, height=0.8, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        left.plot([low * 100, high * 100], [y, y], color=INK, linewidth=0.9, alpha=0.45, zorder=3)
        left.text(high * 100 + 1.0, y, f"{share * 100:.0f}%", va="center", fontsize=9.5, color=INK, zorder=4,
                  fontweight="bold" if kind == "trained" else "normal")
        positions.setdefault(name, []).append(y)
        previous = name
        y += 1.0
    ticks = [sum(ys) / len(ys) for ys in positions.values()]
    labels = list(positions)
    left.set_yticks(ticks)
    left.set_yticklabels(labels, fontsize=10.5)
    left.invert_yaxis()
    left.set_xlim(0, 108)
    left.set_xticks(range(0, 101, 25))
    left.set_xticklabels([f"{v}%" for v in range(0, 101, 25)], fontsize=9.5)
    left.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    left.set_title(f"Naming the bug: top-1 accuracy on the {set_name} ({ours['n']} runs)", loc="left",
                   fontsize=12, fontweight="bold", color=INK, pad=62)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AQUA, label="ours: trained on this task (statistics + code facts)")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=colour, label=f"LLM given {text}") for text, colour in KINDS.values()]
    left.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9, ncol=1)

    # ---- right: accuracy against parameters
    right.set_xscale("log")
    right.scatter([ours["parameters"]], [ours["correct"] / ours["n"] * 100], s=90, color=AQUA, edgecolors=SURFACE,
                  linewidths=1.5, zorder=3)
    right.annotate(f"Ours · {ours['parameters']:,} weights", (ours["parameters"], ours["correct"] / ours["n"] * 100),
                   xytext=(10, 0), textcoords="offset points", va="center", fontsize=9.5, fontweight="bold", color=INK)
    points = []
    for name in names:
        entries = [r for r in measured if r["model"] == name]
        best = max(entries, key=lambda r: r["accuracy"])
        points.append((best["accuracy"] * 100, best["parameters"], name, KINDS[best["kind"]][1]))
    label_y: List[float] = []
    for value, params, name, colour in sorted(points, reverse=True):
        right.scatter([params], [value], s=60, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        y = value + 3.0  # just above the point, moved down until it is clear of the label above it
        for other in label_y:
            if abs(y - other) < 5.0:
                y = other - 5.0
        label_y.append(y)
        right.annotate(f"{name} · {params_text(params)}", (params, value), xytext=(params / 2.2, y),
                       textcoords="data", ha="right", va="center", fontsize=9, color=INK,
                       arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4})
    right.set_xlim(300, 1e12)
    right.set_ylim(0, 105)
    right.set_yticks(range(0, 101, 25))
    right.set_yticklabels([f"{v}%" for v in range(0, 101, 25)], fontsize=9.5)
    right.set_xticks([1e3, 1e6, 1e9, 1e12])
    right.set_xticklabels(["1 thousand", "1 million", "1 billion", "1 trillion"], fontsize=9.5)
    right.minorticks_off()
    right.grid(color=GRID, linewidth=0.8, zorder=0)
    right.set_xlabel("Parameters (log scale)", color=MUTED, fontsize=10)
    right.set_title("Accuracy against size (each LLM at its better input)", loc="left", fontsize=12,
                    fontweight="bold", color=INK, pad=62)

    fig.suptitle("A 1,178-weight model against public LLMs at naming the bug in a PyTorch training run",
                 x=0.012, y=0.995, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    notes = (f"{set_name}: {ours['n']} runs of 6 scripts from Google Colab notebooks (PyTorch tutorials, d2l.ai) on "
             "Colab's sample data, bugs planted by hand, never used to choose anything. Nine silent bugs or \"clean\"."
             " Each LLM got the same catalogue of the ten labels with a one-line description each and answered in JSON;"
             " given the script only, and given the script plus the 62 named statistics our model reads. LLMs via"
             " Groq's free tier (temperature 0.2); parameter counts as published by their makers. Thin lines: 95%"
             " Wilson intervals. Code: examples/compare_llms.py, plot_llms.py.")
    if missing:
        notes += " Not measured (did not answer every run): " + ", ".join(missing) + "."
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(notes, 210)), fontsize=8.2, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
