"""One chart of every version, Shreyas's included, on the same scripts.

    python examples/compare_classifier_versions.py --data traces_v4.jsonl --v2-data traces_v2.jsonl \
        --v4-data traces_v4_150.jsonl --json classifier_versions.json
    python examples/compare_versions.py --version shreyas=../shreyas --version v1=../v1 --version v3=../v3 \
        --version v4=. --scripts old colab --data traces_v4_150.jsonl --json versions_all.json
    python examples/plot_all_versions.py classifier_versions.json versions_all.json all_versions.png

Left: naming the bug in a run. Shreyas's version names it with its rule detectors (the
top cause of its own `fix`); v1 to v4 with the trained trace classifier. Right: the
whole pipeline, offline, on the same scripts (v2 never shipped on its own: it was a
classifier trained between v1 and v3, and its pipeline was v1's). Bars carry their
values; the thin lines are 95% Wilson intervals.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # the reference palette's first three slots
CLASSES = {"clean", "missing_zero_grad", "train_eval_mode_mismatch", "softmax_before_cross_entropy",
           "wrong_loss_reduction", "learning_rate_misconfigured", "frozen_layer_no_grad", "dead_relu",
           "non_normalized_inputs", "label_data_misalignment"}
OLD_SETS = ("benchmarks", "hand-written test")
NAMES = {"shreyas": "v0\n(original)", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4\n(this one)"}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def shreyas_naming(pipeline: Dict[str, Any], sets: Sequence[str]) -> Tuple[int, int]:
    """Shreyas's version naming the bug: its top cause for a buggy run, no cause at all for a healthy one."""
    rows = [r for r in pipeline["rows"] if r["set"] in sets and r["label"] in CLASSES and "shreyas" in r]
    correct = sum((r["shreyas"].get("top_cause") or "clean") == r["label"] for r in rows)
    return correct, len(rows)


def bars(ax: Any, groups: List[str], series: List[Tuple[str, str, List[Tuple[int, int]]]], title: str,
         note_below: Dict[str, str] = None) -> None:
    width = 0.8 / len(series)
    for s, (label, colour, values) in enumerate(series):
        for g, (k, n) in enumerate(values):
            if not n:
                continue
            x = g - 0.4 + width * (s + 0.5)
            share = k / n
            ax.bar(x, share * 100, width=width - 0.03, color=colour, edgecolor=SURFACE, linewidth=1.5,
                   label=label if g == next(i for i, v in enumerate(values) if v[1]) else None, zorder=2)
            low, high = wilson(k, n)
            ax.plot([x, x], [low * 100, high * 100], color=INK, linewidth=0.9, alpha=0.55, zorder=3)
            ax.text(x, share * 100 + 1.2, f"{share * 100:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                    zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([NAMES.get(g, g) + (f"\n{note_below[g]}" if note_below and g in note_below else "")
                        for g in groups], fontsize=9.5, color=INK)
    ax.set_ylim(0, 136)  # the same scale in both panels, with room above the bars for the legend
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.8, ncol=1, bbox_to_anchor=(0.0, 1.02))


def main(classifier_file: str, pipeline_file: str, out: str) -> int:
    classifier = json.loads(Path(classifier_file).read_text(encoding="utf-8"))
    pipeline = json.loads(Path(pipeline_file).read_text(encoding="utf-8"))
    names = ["shreyas", "v1", "v2", "v3", "v4"]
    colab, old = [], []
    for name in names:
        if name == "shreyas":
            colab.append(shreyas_naming(pipeline, ("Colab test",)))
            old.append(shreyas_naming(pipeline, OLD_SETS))
        else:
            got = classifier["versions"][name]
            colab.append((got["Colab test"]["correct"], got["Colab test"]["n"]))
            old.append((got["old test"]["correct"], got["old test"]["n"]))

    versions = [n for n in ("shreyas", "v1", "v3", "v4") if n in pipeline["summary"]["all"]]
    summary = pipeline["summary"]["all"]
    first = [(summary[v]["right_cause_first"], summary[v]["buggy"]) for v in versions]
    fixed = [(summary[v]["planted_bug_fixed"], summary[v]["buggy"]) for v in versions]
    left_alone = [(summary[v]["healthy"] - summary[v]["healthy_changed"], summary[v]["healthy"]) for v in versions]
    times = {v: f"{summary[v]['median_seconds_buggy']:.0f} s per buggy script" for v in versions}

    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 6.6), gridspec_kw={"width_ratios": [5, 4.4]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
    n_colab, n_old = colab[-1][1], old[-1][1]
    bars(left, names, [(f"Colab test: {n_colab} runs from Colab notebooks, never tuned on", BLUE, colab),
                       (f"Old test: {n_old} hand-written runs (seen while tuning v3 and v4)", ORANGE, old)],
         "Naming the bug in a training run (top-1 accuracy)",
         note_below={"shreyas": "rule detectors", "v1": "classifier", "v2": "classifier", "v3": "classifier",
                     "v4": "classifier"})
    n_buggy, n_healthy = summary[versions[0]]["buggy"], summary[versions[0]]["healthy"]
    bars(right, versions, [(f"Right cause ranked first (of {n_buggy} buggy scripts)", BLUE, first),
                           (f"Planted bug fixed and validated (of {n_buggy})", ORANGE, fixed),
                           (f"Healthy scripts left alone (of {n_healthy})", AQUA, left_alone)],
         "The whole pipeline, offline (fix, no LLM)", note_below=times)
    fig.suptitle("debug101, every version on the same scripts", x=0.012, y=0.995, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    notes = ("v0 is Shreyas's original version (DLDebugger, commit 0619721). "
             "Left: v0 names the bug with its rule detectors (top cause of its own fix); v1-v4 with "
             "the trained trace classifier. v2 was never released on its own (a classifier between v1 and v3, "
             "rebuilt exactly from its data), so it has no pipeline bar. "
             f"Right: {n_buggy + n_healthy} scripts (the old test scripts, the project benchmarks including bugs "
             "outside the classifier's nine, and the Colab test), each version's own fix --dry-run --no-llm, run "
             "back to back on one machine (Apple M2 CPU). Thin lines: 95% Wilson intervals. "
             "Code: examples/compare_versions.py, compare_classifier_versions.py, plot_all_versions.py.")
    import textwrap

    fig.text(0.012, 0.01, "\n".join(textwrap.wrap(notes, 205)), fontsize=8.2, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
