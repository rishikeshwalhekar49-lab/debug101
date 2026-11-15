"""Lightweight NDJSON event emitter for the Pastel + Ink TUI.

When a command is invoked with ``--events``, every agent phase, finding,
and state change is emitted as a newline-delimited JSON object on stdout so
the Ink frontend can render live spinners, progress meters, and diff views.

Usage::

    from .events import EventEmitter
    emitter = EventEmitter(enabled=args.events)
    emitter.phase("perception", "start")
    emitter.finding(finding_dict)
    emitter.phase("perception", "done", findings=3)
    emitter.result(report_dict)
    emitter.error("something went wrong")

When ``enabled=False`` the emitter is a no-op so normal text output is
unaffected.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional


class EventEmitter:
    """Emits NDJSON events to stdout (or a custom stream)."""

    def __init__(self, enabled: bool = False, stream=None) -> None:
        self.enabled = enabled
        self._stream = stream or sys.stdout

    # ------------------------------------------------------------------
    def _emit(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload.setdefault("ts", round(time.time() * 1000))
        line = json.dumps(payload, default=str)
        self._stream.write(line + "\n")
        self._stream.flush()

    # ------------------------------------------------------------------
    def phase(self, name: str, status: str, **extra: Any) -> None:
        """Emit a pipeline phase lifecycle event.

        ``status`` is one of ``"start"``, ``"done"``, ``"error"``.
        """
        self._emit({"event": "phase", "phase": name, "status": status, **extra})

    def finding(self, finding: Dict[str, Any]) -> None:
        """Emit a single Finding (as a dict) so the UI can stream results."""
        self._emit({"event": "finding", "data": finding})

    def findings_batch(self, findings: List[Dict[str, Any]], diagnostics: Optional[List[str]] = None) -> None:
        """Emit an entire Report's findings at once."""
        self._emit({
            "event": "findings_batch",
            "findings": findings,
            "diagnostics": diagnostics or [],
        })

    def hypothesis(self, rank: int, bug_id: str, confidence: float,
                   description: str, diagnosed_by: Optional[str] = None) -> None:
        """Emit a ranked diagnostic hypothesis."""
        self._emit({
            "event": "hypothesis",
            "rank": rank,
            "bug_id": bug_id,
            "confidence": confidence,
            "description": description,
            "diagnosed_by": diagnosed_by,
        })

    def patch(self, bug_id: str, strategy: str, source: str,
              diff: str, confidence: float) -> None:
        """Emit a generated patch candidate. It has not been validated yet; see ``accepted_patch``."""
        self._emit({
            "event": "patch",
            "bug_id": bug_id,
            "strategy": strategy,
            "source": source,
            "diff": diff,
            "confidence": confidence,
        })

    def verdict(self, bug_id: str, accepted: bool, validator: str,
                p_value: Optional[float], effect_size: Optional[float],
                notes: str) -> None:
        """Emit a validation verdict."""
        self._emit({
            "event": "verdict",
            "bug_id": bug_id,
            "accepted": accepted,
            "validator": validator,
            "p_value": p_value,
            "effect_size": effect_size,
            "notes": notes,
        })

    def accepted_patch(self, bug_id: str, strategy: str, source: str, diff: str, confidence: float, *,
                       applied: bool, backup: Optional[str] = None, pending: bool = False,
                       blocked: Optional[str] = None) -> None:
        """Emit the one patch that passed validation, and what happened to the file.

        ``applied`` is true only if the script was written. ``pending`` means the patch is
        kept for the ``apply`` command. ``blocked`` gives the reason it cannot be applied."""
        self._emit({
            "event": "accepted_patch",
            "bug_id": bug_id,
            "strategy": strategy,
            "source": source,
            "diff": diff,
            "confidence": confidence,
            "applied": applied,
            "backup": backup,
            "pending": pending,
            "blocked": blocked,
        })

    def applied(self, target: str, backup: Optional[str], bug_id: str = "") -> None:
        """Emit that ``apply`` wrote a pending patch to the script."""
        self._emit({"event": "applied", "target": target, "backup": backup, "bug_id": bug_id})

    def weight_report(self, data: Dict[str, Any]) -> None:
        """Emit a read-only weight inspection report."""
        self._emit({"event": "weight_report", "data": data})

    def result(self, data: Dict[str, Any]) -> None:
        """Emit the final aggregated result object."""
        self._emit({"event": "result", "data": data})

    def doctor_check(self, ok: bool, label: str, advice: str = "") -> None:
        """Emit a single doctor health-check result."""
        self._emit({"event": "doctor_check", "ok": ok, "label": label, "advice": advice})

    def runtime_alert(self, severity: str, bug_id: str,
                      file: str, line: int, message: str) -> None:
        """Emit a live runtime hook alert."""
        self._emit({
            "event": "runtime_alert",
            "severity": severity,
            "bug_id": bug_id,
            "file": file,
            "line": line,
            "message": message,
        })

    def error(self, message: str, detail: str = "") -> None:
        """Emit a fatal error event."""
        self._emit({"event": "error", "message": message, "detail": detail})

    def done(self, exit_code: int = 0) -> None:
        """Emit the final done sentinel so the UI knows the stream ended cleanly."""
        self._emit({"event": "done", "exit_code": exit_code})

