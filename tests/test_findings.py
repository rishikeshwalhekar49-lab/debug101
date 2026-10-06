"""Tests for silent_bug_agent.findings (Finding, Report, merge_reports)."""

import json
from pathlib import Path
import pytest

from silent_bug_agent.findings import Finding, Report, merge_reports, parse_severity
from silent_bug_agent.invariants.base import Severity


def test_parse_severity():
    assert parse_severity("info") == Severity.INFO
    assert parse_severity("WARNING") == Severity.WARNING
    assert parse_severity("critical") == Severity.CRITICAL
    assert parse_severity(Severity.CRITICAL) == Severity.CRITICAL
    with pytest.raises(ValueError):
        parse_severity("invalid_severity")


def test_finding_creation_and_dict_roundtrip():
    f = Finding(
        bug_id="missing_zero_grad",
        detector="static",
        message="optimizer.zero_grad() is missing in the training loop",
        severity=Severity.CRITICAL,
        confidence=0.95,
        file="train.py",
        line=42,
        evidence="loss.backward()",
        suggested_fix="Call optimizer.zero_grad() before backward().",
        rule="MissingZeroGradRule",
        extra={"step": 10},
    )
    assert f.key() == ("missing_zero_grad", "train.py", 42)

    d = f.to_dict()
    assert d["severity"] == "critical"
    assert d["bug_id"] == "missing_zero_grad"
    assert d["line"] == 42
    assert d["extra"]["step"] == 10

    f2 = Finding.from_dict(d)
    assert f2.bug_id == f.bug_id
    assert f2.detector == f.detector
    assert f2.message == f.message
    assert f2.severity == f.severity
    assert f2.confidence == f.confidence
    assert f2.file == f.file
    assert f2.line == f.line
    assert f2.evidence == f.evidence
    assert f2.suggested_fix == f.suggested_fix
    assert f2.rule == f.rule
    assert f2.extra == f.extra


def test_report_sorting_and_filtering():
    f_info = Finding(
        bug_id="wrong_loss_reduction",
        detector="static",
        message="check reduction",
        severity=Severity.INFO,
        confidence=0.5,
        file="model.py",
        line=20,
    )
    f_warn = Finding(
        bug_id="non_normalized_inputs",
        detector="static",
        message="missing norm",
        severity=Severity.WARNING,
        confidence=0.8,
        file="dataset.py",
        line=15,
    )
    f_crit = Finding(
        bug_id="missing_zero_grad",
        detector="static",
        message="no zero_grad",
        severity=Severity.CRITICAL,
        confidence=0.99,
        file="dataset.py",
        line=10,
    )

    report = Report(findings=[f_info, f_warn, f_crit], files_scanned=["model.py", "dataset.py"])

    # sorted by file, line, -severity rank
    sorted_f = report.sorted_findings()
    assert sorted_f[0] == f_crit  # dataset.py:10
    assert sorted_f[1] == f_warn  # dataset.py:15
    assert sorted_f[2] == f_info  # model.py:20

    # at_least filter
    crit_only = report.at_least(Severity.CRITICAL)
    assert len(crit_only) == 1
    assert crit_only[0] == f_crit

    warn_and_above = report.at_least(Severity.WARNING)
    assert len(warn_and_above) == 2
    assert f_info not in warn_and_above

    all_findings = report.at_least(Severity.INFO)
    assert len(all_findings) == 3


def test_report_summary_and_json_roundtrip():
    f1 = Finding(
        bug_id="missing_zero_grad",
        detector="static",
        message="no zero_grad",
        severity=Severity.CRITICAL,
        confidence=1.0,
        file="train.py",
        line=5,
    )
    f2 = Finding(
        bug_id="missing_zero_grad",
        detector="semantic",
        message="zero grad omitted",
        severity=Severity.CRITICAL,
        confidence=0.8,
        file="other.py",
        line=12,
    )
    report = Report(
        findings=[f1, f2],
        files_scanned=["train.py", "other.py"],
        diagnostics=["LLM returned without error"],
    )

    summary = report.summary()
    assert summary["files_scanned"] == 2
    assert summary["total_findings"] == 2
    assert summary["by_severity"]["critical"] == 2
    assert summary["by_bug_id"]["missing_zero_grad"] == 2

    json_str = report.to_json()
    loaded = Report.from_json(json_str)
    assert len(loaded.findings) == 2
    assert loaded.files_scanned == ["train.py", "other.py"]
    assert loaded.diagnostics == ["LLM returned without error"]
    assert loaded.findings[0].bug_id == "missing_zero_grad"


def test_report_extend():
    r1 = Report(
        findings=[
            Finding(
                bug_id="missing_zero_grad",
                detector="static",
                message="m1",
                severity=Severity.CRITICAL,
                confidence=1.0,
            )
        ],
        files_scanned=["a.py"],
        diagnostics=["diag1"],
    )
    r2 = Report(
        findings=[
            Finding(
                bug_id="dead_relu",
                detector="runtime",
                message="m2",
                severity=Severity.WARNING,
                confidence=0.9,
            )
        ],
        files_scanned=["a.py", "b.py"],
        diagnostics=["diag2"],
    )
    r1.extend(r2)
    assert len(r1.findings) == 2
    assert r1.files_scanned == ["a.py", "b.py"]
    assert r1.diagnostics == ["diag1", "diag2"]


def test_report_format_text():
    f = Finding(
        bug_id="softmax_before_cross_entropy",
        detector="static",
        message="Found softmax right before CrossEntropyLoss",
        severity=Severity.WARNING,
        confidence=0.9,
        file="model.py",
        line=45,
        evidence="return F.softmax(x, dim=1)",
        suggested_fix="Remove softmax",
    )
    report = Report(findings=[f], files_scanned=["model.py"], diagnostics=["syntax ok"])
    text = report.format_text()
    assert "model.py:45  [warning] softmax_before_cross_entropy" in text
    assert "return F.softmax(x, dim=1)" in text
    assert "fix: Remove softmax" in text
    assert "note: syntax ok" in text
    assert "1 finding(s) in 1 file(s)" in text


def test_merge_reports_deduplication():
    # Primary (e.g. static) finding
    f_prim = Finding(
        bug_id="missing_zero_grad",
        detector="static",
        message="static detector spotted missing zero_grad",
        severity=Severity.CRITICAL,
        confidence=1.0,
        file="train.py",
        line=50,
        evidence="loss.backward()",
    )
    primary = Report(findings=[f_prim], files_scanned=["train.py"])

    # Duplicate in secondary (within tolerance of 2 lines)
    f_sec_dup = Finding(
        bug_id="missing_zero_grad",
        detector="semantic",
        message="semantic detector spotted missing zero_grad",
        severity=Severity.CRITICAL,
        confidence=0.85,
        file="train.py",
        line=51,  # within 2 lines
    )
    # Distinct bug in secondary
    f_sec_distinct = Finding(
        bug_id="non_normalized_inputs",
        detector="semantic",
        message="data transforms missing normalization",
        severity=Severity.WARNING,
        confidence=0.75,
        file="train.py",
        line=15,
    )
    secondary = Report(
        findings=[f_sec_dup, f_sec_distinct],
        files_scanned=["train.py", "eval.py"],
        diagnostics=["semantic scan finished"],
    )

    merged = merge_reports(primary, secondary, line_tolerance=2)

    assert len(merged.findings) == 2
    # primary must be preserved as-is
    assert merged.findings[0] == f_prim
    # duplicate secondary was dropped; distinct secondary was kept
    assert merged.findings[1] == f_sec_distinct
    assert "train.py" in merged.files_scanned
    assert "eval.py" in merged.files_scanned
    assert "semantic scan finished" in merged.diagnostics
