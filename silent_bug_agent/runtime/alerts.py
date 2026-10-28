"""Alert aggregation and early-alert dispatch (Runtime Monitor Agent tool).

Signals arrive once per snapshot and per check, so a real bug produces
hundreds of them (``frozen_weights`` fired 181 times on the planted frozen-layer
run). :class:`AlertDispatcher` turns them into **one alert per bug per model**:

* **aggregation** -- every signal for the same ``(channel, bug_id)`` updates one
  :class:`Alert`: first/last step, count per rule, peak value, offenders, and the
  context of the first occurrence (a trace of that moment);
* **confirmation** -- critical signals and checks that already have their own
  patience (windowed invariants, trends) confirm an alert at once; noisy
  per-step checks need ``confirm_after`` occurrences, so a single transient
  reading never becomes a finding;
* **early alerts** -- callbacks run the moment an alert is confirmed (and again
  if it escalates to critical), while training is still running;
* **findings** -- :meth:`AlertDispatcher.findings` builds the
  :class:`~silent_bug_agent.findings.Finding` objects for the final report.

Callbacks run on the agent's consumer thread; an exception in one is counted and
never reaches training.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..findings import SEVERITY_RANK, Finding
from ..invariants.base import Severity
from .anomaly import ALTERNATIVE_BUG_IDS
from .trends import Signal

# Rules that need repeated occurrences before they become an alert (transient per-step readings).
NOISY_RULES = {"invariant:gradient_norm_bounds", "invariant:activation_health", "runtime:loss_spike"}

Locator = Callable[["Alert"], Tuple[str, int, str]]  # alert -> (file, line, evidence source line)


@dataclass
class RuleStats:
    rule: str
    severity: Severity
    base_confidence: float
    count: int = 0
    first_step: int = 0
    last_step: int = 0
    peak: Optional[float] = None
    message: str = ""
    first_message: str = ""

    @property
    def confidence(self) -> float:
        """Grows with persistence: base at one occurrence, half-way to 1.0 after 11."""
        b = self.base_confidence
        return min(0.99, b + (1.0 - b) * 0.5 * min(1.0, (self.count - 1) / 10.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "first_step": self.first_step, "last_step": self.last_step,
                "peak": self.peak, "severity": self.severity.value, "confidence": round(self.confidence, 3),
                "message": self.message}


@dataclass
class Alert:
    channel: str
    bug_id: str
    severity: Severity
    first_step: int
    last_step: int
    phase: str = "train"
    anchor: str = "step"
    layer: Optional[str] = None
    confirmed_step: Optional[int] = None
    rules: Dict[str, RuleStats] = field(default_factory=dict)
    offenders: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # trace of the first occurrence
    details: Dict[str, Any] = field(default_factory=dict)
    snapshots_seen: int = 0  # monitored snapshots of this channel since first_step

    @property
    def confirmed(self) -> bool:
        return self.confirmed_step is not None

    @property
    def confidence(self) -> float:
        """Noisy-OR over the confirmed rules that support this bug."""
        p = 1.0
        for r in self._supporting():
            p *= 1.0 - r.confidence
        return min(0.99, 1.0 - p)

    @property
    def occurrences(self) -> int:
        return max((r.count for r in self.rules.values()), default=0)

    def _supporting(self) -> List[RuleStats]:
        rs = [r for r in self.rules.values() if r.count >= _confirm_after(r.rule, r.severity)]
        return rs or list(self.rules.values())

    def primary(self) -> RuleStats:
        return max(self._supporting(), key=lambda r: (SEVERITY_RANK[r.severity], r.confidence, r.count))

    def message(self, show_channel: bool = False) -> str:
        p = self.primary()
        span = f"step {self.first_step}" if self.first_step == self.last_step else f"steps {self.first_step}-{self.last_step}"
        persist = f"{self.occurrences} of {max(self.snapshots_seen, self.occurrences)} monitored snapshots"
        msg = f"[runtime, {span}, {persist}] {p.first_message}"
        others = [r for r in self._supporting() if r is not p]
        if others:
            msg += " Also: " + "; ".join(f"{r.rule} x{r.count}" for r in others) + "."
        if show_channel:
            msg = f"({self.channel}) " + msg
        return msg

    def to_finding(self, locator: Optional[Locator] = None, file: str = "<runtime>", show_channel: bool = False) -> Finding:
        path, line, evidence = (locator(self) if locator else (file, 0, ""))
        return Finding(
            bug_id=self.bug_id,
            detector="runtime",
            message=self.message(show_channel),
            severity=self.severity,
            confidence=round(self.confidence, 3),
            file=path,
            line=line,
            evidence=evidence,
            rule=self.primary().rule,
            extra={
                "channel": self.channel,
                "step": self.first_step,
                "first_step": self.first_step,
                "last_step": self.last_step,
                "confirmed_step": self.confirmed_step,
                "occurrences": self.occurrences,
                "monitored_snapshots": max(self.snapshots_seen, self.occurrences),
                "phase": self.phase,
                "rules": {k: r.to_dict() for k, r in self.rules.items()},
                "offenders": dict(list(self.offenders.items())[:10]),
                "alternative_bug_ids": ALTERNATIVE_BUG_IDS.get(self.bug_id, []),
                "context": self.context,
                **({"details": self.details} if self.details else {}),
            },
        )


def _confirm_after(rule: str, severity: Severity, noisy: int = 2) -> int:
    if severity == Severity.CRITICAL or rule not in NOISY_RULES:
        return 1
    return noisy


class AlertDispatcher:
    def __init__(self, *, confirm_after: int = 2, min_severity: Severity = Severity.WARNING) -> None:
        self.confirm_after = confirm_after
        self.min_severity = min_severity
        self.alerts: Dict[Tuple[str, str], Alert] = {}
        self.events: List[Tuple[str, int, str, str]] = []  # (kind, step, channel, bug_id): "new" | "escalated"
        self.callback_errors = 0
        self._callbacks: List[Callable[[Alert, str], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Alert, str], None]) -> Callable[[], None]:
        """``callback(alert, kind)`` with kind "new" or "escalated"; returns an unsubscribe function."""
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def tick(self, channel: str, phase: str = "train") -> None:
        """One more monitored ``phase`` snapshot of ``channel`` (for the persistence ratio)."""
        for a in self.alerts.values():
            if a.channel == channel and a.phase == phase:
                a.snapshots_seen += 1

    def ingest(self, channel: str, signals: List[Signal],
               context: Optional[Callable[[Signal], Dict[str, Any]]] = None) -> List[Alert]:
        """Fold this snapshot's signals into alerts; returns alerts that were confirmed or escalated now.

        ``context(signal)`` builds the trace stored with a *new* alert (called once per alert)."""
        fired: List[Tuple[Alert, str]] = []
        for s in signals:
            if SEVERITY_RANK[s.severity] < SEVERITY_RANK[self.min_severity]:
                continue
            key = (channel, s.bug_id)
            a = self.alerts.get(key)
            if a is None:
                a = Alert(channel, s.bug_id, s.severity, s.step, s.step, phase=s.phase, anchor=s.anchor,
                          layer=s.layer, context=context(s) if context else {}, snapshots_seen=1)
                self.alerts[key] = a
            r = a.rules.get(s.rule)
            if r is None:
                r = a.rules[s.rule] = RuleStats(s.rule, s.severity, s.confidence, first_step=s.step,
                                                first_message=s.message)
            if r.count == 0 or s.step != r.last_step:  # one count per rule per snapshot (several layers may fire)
                r.count += 1
            r.last_step = s.step
            r.message = s.message
            if SEVERITY_RANK[s.severity] > SEVERITY_RANK[r.severity]:
                r.severity = s.severity
            if s.value is not None and s.value == s.value:
                r.peak = s.value if r.peak is None else max(r.peak, s.value)
            a.last_step = max(a.last_step, s.step)
            for k, v in s.offenders.items():
                if k not in a.offenders or (v == v and abs(v) > abs(a.offenders[k])):
                    a.offenders[k] = v
            if s.details:
                a.details.update(s.details)
            if r.count < _confirm_after(s.rule, r.severity, self.confirm_after):
                continue
            # confirmed rules decide the alert's severity and where it points in the code
            if not a.confirmed:
                a.confirmed_step = s.step
                a.severity, a.anchor, a.layer = r.severity, s.anchor, s.layer or a.layer
                fired.append((a, "new"))
            elif SEVERITY_RANK[r.severity] > SEVERITY_RANK[a.severity]:
                a.severity, a.anchor, a.layer = r.severity, s.anchor, s.layer or a.layer
                fired.append((a, "escalated"))
        out = []
        for a, kind in fired:
            self.events.append((kind, a.last_step, a.channel, a.bug_id))
            self._dispatch(a, kind)
            out.append(a)
        return out

    def _dispatch(self, alert: Alert, kind: str) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(alert, kind)
            except Exception:  # a broken consumer must never break monitoring
                self.callback_errors += 1

    @property
    def suppressed(self) -> int:
        """Signals that never became a finding (transient readings below ``confirm_after``)."""
        return sum(r.count for a in self.unconfirmed() for r in a.rules.values())

    def confirmed(self) -> List[Alert]:
        return [a for a in self.alerts.values() if a.confirmed]

    def unconfirmed(self) -> List[Alert]:
        return [a for a in self.alerts.values() if not a.confirmed]

    def findings(self, locator: Optional[Locator] = None, file: str = "<runtime>",
                 show_channel: bool = False) -> List[Finding]:
        out = [a.to_finding(locator, file, show_channel) for a in self.confirmed()]
        out.sort(key=lambda f: (-SEVERITY_RANK[f.severity], -f.confidence, f.extra.get("first_step", 0)))
        return out
