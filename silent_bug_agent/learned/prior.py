"""How the trace classifier takes part in a run.

During perception the script is run once more, traced, at the same time as the
runtime monitor (so it costs little extra time). What happens next depends on the mode:

* "advisory" (the default): the guess is shown in the report, and nothing else.
* "active": the probabilities also take part. A confident guess (p >= FINDING_THRESHOLD)
  of a bug no detector reported becomes a runtime finding, the probabilities nudge the
  offline ranking of the hypotheses (by at most WEIGHT), and an LLM is given them as
  one more piece of evidence.
* "off": no traced run at all.

Advisory is the default because "active" was measured and made results worse: on the
blind benchmarks it repaired 8 of 18 instead of 9 and ranked the right cause first for
13 instead of 14 (docs/learned_classifier.md). Every patch still has to pass both
validation gates, whatever the classifier said.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..findings import Finding, Severity

FINDING_THRESHOLD = 0.9  # held-out templates: 3.8% of healthy runs reach it wrongly, 83% of bugs are caught
WEIGHT = 0.3
CLEAN = "clean"
MODES = ("off", "advisory", "active")


def predict_script(path: Path, *, script_args: Sequence[str] = (), timeout: float = 120.0,
                   max_steps: int = 60) -> Optional[Dict[str, Any]]:
    """Trace the script once and classify the trace; None when there is no classifier or the run failed."""
    from ..mutation.dataset import trace_run, usable
    from .model import load_default

    model = load_default()
    if model is None:
        return None
    started = time.perf_counter()
    path = Path(path).resolve()
    try:
        code = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    report = trace_run(code, max_steps=max_steps, timeout=timeout, cwd=str(path.parent), as_file=str(path),
                       script_args=list(script_args))
    if not usable(report):
        return None
    probabilities = model.predict_report(report)
    top, p = next(iter(probabilities.items()))
    return {"top": top, "p": p, "probabilities": dict(list(probabilities.items())[:5]),
            "seconds": round(time.perf_counter() - started, 2),
            "model": {"benchmark_accuracy": (model.meta.get("benchmarks") or {}).get("accuracy"),
                      "trained_on_runs": model.meta.get("trained_on_runs")}}


def finding_from(prediction: Optional[Dict[str, Any]], file: str, known: Sequence[str]) -> Optional[Finding]:
    """A finding for a confident guess that no detector made."""
    if not prediction or prediction["top"] == CLEAN or prediction["p"] < FINDING_THRESHOLD:
        return None
    if prediction["top"] in known:
        return None
    return Finding(bug_id=prediction["top"], detector="runtime", rule="trace_classifier", severity=Severity.WARNING,
                   confidence=round(min(0.9, prediction["p"]), 3), file=file, line=0,
                   message=(f"The training trace looks like {prediction['top']} to the trained trace classifier "
                            f"(p={prediction['p']:.2f}). This is a learned guess from the run's statistics, "
                            "not a rule match."),
                   extra={"learned": True, "probabilities": prediction["probabilities"]})


def apply_prior(hypotheses: List[Any], prediction: Optional[Dict[str, Any]], weight: float = WEIGHT) -> List[Any]:
    """Re-rank hypotheses by confidence plus `weight` times the classifier's probability for that bug."""
    if not prediction or not hypotheses:
        return hypotheses
    probabilities = prediction["probabilities"]
    for h in hypotheses:
        if h.bug_id in probabilities:
            h.extra["learned_probability"] = probabilities[h.bug_id]
    return sorted(hypotheses, key=lambda h: -(h.confidence + weight * probabilities.get(h.bug_id, 0.0)))


def prompt_line(prediction: Optional[Dict[str, Any]]) -> str:
    if not prediction:
        return ""
    shown = ", ".join(f"{bug} {p:.2f}" for bug, p in prediction["probabilities"].items())
    return ("- trained trace classifier (learned from generated buggy scripts; treat it as a hint, not proof): "
            + shown)
