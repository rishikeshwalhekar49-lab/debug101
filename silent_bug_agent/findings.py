"""Common bug-report format shared by every detector.

Static analysis (Module 2), semantic analysis (Module 2) and, later, the
runtime monitor (Module 1) all emit `Finding` objects, so whatever
consumes them (the supervisor, a CLI, a CI job) never needs to know
which detector produced what. `bug_id` always refers to an entry in the
Module 0 taxonomy, which is what ties the detectors together.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .invariants.base import Severity

SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


def parse_severity(value) -> Severity:
    if isinstance(value, Severity):
        return value
    return Severity(str(value).lower())


@dataclass
class Finding:
    bug_id: str  # id of a Module 0 taxonomy entry
    detector: str  # "static" | "semantic" | "runtime"
    message: str
    severity: Severity
    confidence: float  # 0..1, detector's own estimate that this is a real bug
    file: str = "<unknown>"
    line: int = 0
    evidence: str = ""  # the actual source line(s), grounded in the file
    suggested_fix: str = ""  # taken from the taxonomy's typical_fix
    rule: str = ""  # which rule / reviewer produced it
    extra: Dict[str, Any] = field(default_factory=dict)

    def key(self):
        return (self.bug_id, self.file, self.line)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Finding":
        d = dict(d)
        d["severity"] = parse_severity(d["severity"])
        return cls(**d)


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    files_scanned: List[str] = field(default_factory=list)
    # Non-fatal problems: syntax errors, LLM failures, dropped model output.
    # A detector should report these here rather than raise, so one bad
    # file never aborts a whole scan.
    diagnostics: List[str] = field(default_factory=list)

    def extend(self, other: "Report") -> None:
        self.findings.extend(other.findings)
        for f in other.files_scanned:
            if f not in self.files_scanned:
                self.files_scanned.append(f)
        self.diagnostics.extend(other.diagnostics)

    def sorted_findings(self) -> List[Finding]:
        return sorted(
            self.findings,
            key=lambda f: (f.file, f.line, -SEVERITY_RANK[f.severity]),
        )

    def at_least(self, severity: Severity) -> List[Finding]:
        floor = SEVERITY_RANK[severity]
        return [f for f in self.findings if SEVERITY_RANK[f.severity] >= floor]

    def summary(self) -> Dict[str, Any]:
        by_sev: Dict[str, int] = {}
        by_bug: Dict[str, int] = {}
        for f in self.findings:
            by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1
            by_bug[f.bug_id] = by_bug.get(f.bug_id, 0) + 1
        return {
            "files_scanned": len(self.files_scanned),
            "total_findings": len(self.findings),
            "by_severity": by_sev,
            "by_bug_id": by_bug,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.sorted_findings()],
            "files_scanned": list(self.files_scanned),
            "diagnostics": list(self.diagnostics),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Report":
        d = json.loads(text)
        return cls(
            findings=[Finding.from_dict(f) for f in d.get("findings", [])],
            files_scanned=list(d.get("files_scanned", [])),
            diagnostics=list(d.get("diagnostics", [])),
        )

    def format_text(self, min_severity: Severity = Severity.INFO) -> str:
        floor = SEVERITY_RANK[min_severity]
        out: List[str] = []
        for f in self.sorted_findings():
            if SEVERITY_RANK[f.severity] < floor:
                continue
            out.append(
                f"{f.file}:{f.line}  [{f.severity.value}] {f.bug_id}  "
                f"({f.detector}, confidence {f.confidence:.2f})"
            )
            out.append(f"    {f.message}")
            if f.evidence:
                out.append(f"    > {f.evidence.strip()}")
            if f.suggested_fix:
                out.append(f"    fix: {f.suggested_fix}")
        for d in self.diagnostics:
            out.append(f"note: {d}")
        s = self.summary()
        out.append(
            f"\n{s['total_findings']} finding(s) in {s['files_scanned']} file(s) "
            f"{s['by_severity'] or ''}".rstrip()
        )
        return "\n".join(out)


def merge_reports(primary: Report, secondary: Report, line_tolerance: int = 2) -> Report:
    """Combine two reports, dropping `secondary` findings that duplicate `primary`.

    Two findings are duplicates when they name the same bug in the same
    file within `line_tolerance` lines of each other. `primary` wins, so
    pass the static report as primary: its findings are deterministic
    and grounded, while a semantic finding at the same spot adds nothing.
    """
    merged = Report()
    merged.extend(primary)
    for f in secondary.findings:
        duplicate = any(
            p.bug_id == f.bug_id and p.file == f.file and abs(p.line - f.line) <= line_tolerance
            for p in primary.findings
        )
        if not duplicate:
            merged.findings.append(f)
    for name in secondary.files_scanned:
        if name not in merged.files_scanned:
            merged.files_scanned.append(name)
    merged.diagnostics.extend(secondary.diagnostics)
    return merged
