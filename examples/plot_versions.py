"""Draw the version comparison from `compare_versions.py --json` as a terminal-style table.

    python examples/plot_versions.py versions.json versions.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_leaderboard import draw_table  # noqa: E402

CAPABILITIES = [
    ("Pipeline sees a blind copy (no hints)", "no", "yes"),
    ("Gate 2 seeds in parallel, two looks", "no", "yes"),
    ("Time per stage reported", "no", "yes"),
    ("Code snippets (snippet)", "no", "yes"),
    ("Trained trace classifier", "no", "yes, advisory"),
]


def change(old: float, new: float, lower_is_better: bool = False) -> str:
    if old == new:
        return "same"
    better = (new < old) if lower_is_better else (new > old)
    return f"{'better' if better else 'worse'} ({new - old:+g})"


def rows_for(summary: Dict[str, Any]) -> List[List[str]]:
    old, new = summary["old"], summary["new"]
    rows = []
    extra_repairs = (new["repaired"] - old["repaired"]) - (new["planted_bug_fixed"] - old["planted_bug_fixed"])
    for label, key, lower in (("Buggy scripts repaired", "repaired", False),
                              ("Right cause ranked first", "right_cause_first", False),
                              ("Planted bug itself fixed", "planted_bug_fixed", False)):
        verdict = change(old[key], new[key], lower)
        if key == "repaired" and extra_repairs > 0:
            verdict = f"{new[key] - old[key]:+d}, a symptom fix"
        rows.append([label, f"{old[key]} of {old['buggy']}", f"{new[key]} of {new['buggy']}", verdict])
    rows.append(["Healthy scripts changed (lower is better)", f"{old['healthy_changed']} of {old['healthy']}",
                 f"{new['healthy_changed']} of {new['healthy']}",
                 change(old["healthy_changed"], new["healthy_changed"], True)])
    rows.append(["Median time per buggy script", f"{old['median_seconds_buggy']:.0f} s",
                 f"{new['median_seconds_buggy']:.0f} s",
                 f"{old['median_seconds_buggy'] / new['median_seconds_buggy']:.1f}x faster"
                 if new["median_seconds_buggy"] < old["median_seconds_buggy"] else "slower"])
    rows.append(["Total time, all scripts", f"{old['total_seconds'] / 60:.0f} min", f"{new['total_seconds'] / 60:.0f} min",
                 f"{old['total_seconds'] / new['total_seconds']:.1f}x faster"
                 if new["total_seconds"] < old["total_seconds"] else "slower"])
    rows.append(["Runs that crashed", str(old["errors"]), str(new["errors"]), change(old["errors"], new["errors"], True)])
    return rows


def differences(data: Dict[str, Any]) -> List[str]:
    """One line for every script on which the two versions decided differently."""
    lines = []
    for row in data["rows"]:
        old, new = row["old"], row["new"]
        if (old.get("patched"), old.get("fixed_bug")) == (new.get("patched"), new.get("fixed_bug")):
            continue
        def said(v: Dict[str, Any]) -> str:
            if not v.get("patched"):
                return "no fix"
            return "fixed the planted bug" if v.get("fixed_bug") == row["label"] else f"fixed a symptom ({v.get('fixed_bug')})"
        lines.append(f"Only difference: {row['name']}: Shreyas's version {said(old)}, the new one {said(new)}.")
    return lines


def main(source: str, out: str) -> int:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    everything = data["summary"]["all"]
    n = everything["old"]["buggy"] + everything["old"]["healthy"]
    cols = [("Measure", 30), ("Shreyas's version (DLDebugger, 0619721)", 20), ("New version (debug101)", 20),
            ("Change", 18)]
    rows = rows_for(everything) + [[name, old, new, ""] for name, old, new in CAPABILITIES]
    per_set = []
    for subset in ("benchmarks", "hand-written test"):
        o, w = data["summary"][subset]["old"], data["summary"][subset]["new"]
        per_set.append(f"{subset}: repaired {o['repaired']} -> {w['repaired']} of {o['buggy']}, right cause first "
                       f"{o['right_cause_first']} -> {w['right_cause_first']}, healthy changed "
                       f"{o['healthy_changed']} -> {w['healthy_changed']} of {o['healthy']}")
    notes = [f"Same {n} scripts for both, offline (no LLM key), each version's own `fix --dry-run --no-llm`, fresh"
             " memory every run, run back to back on one machine (Apple M2 CPU, 25 Nov 2026).",
             "Scripts: blind copies of the 18 project benchmarks and their healthy twins, and the 58-run hand-written"
             " test set (examples/holdout, scripts 05-10).", *per_set, *differences(data),
             "Code: examples/compare_versions.py, github.com/rishikeshwalhekar49-lab/debug101"]
    draw_table("2. Shreyas's version against the new version: the same scripts, offline", cols, rows, notes, out,
               bold=lambda row: False, dim=lambda row: row[3] == "")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
