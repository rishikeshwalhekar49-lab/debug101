"""Tests for semantic analyzer (FakeLLMClient, grounding, prompt construction, error handling)."""

import json
from pathlib import Path
import pytest

from silent_bug_agent.findings import Finding, Report, merge_reports
from silent_bug_agent.invariants.base import Severity
from silent_bug_agent.semantic import (
    FakeLLMClient,
    SemanticAnalyzer,
    build_analysis_prompt,
    format_semantic_catalog,
)
from silent_bug_agent.taxonomy.loader import load_taxonomy


def test_format_semantic_catalog():
    patterns = load_taxonomy()
    catalog = format_semantic_catalog(patterns)
    assert 'bug_id: "label_data_misalignment"' in catalog
    assert 'bug_id: "data_leakage_train_test"' in catalog
    assert 'bug_id: "mismatched_loss_task"' in catalog


def test_build_analysis_prompt_contains_injection_guards_and_catalog():
    patterns = load_taxonomy()
    source = "def train():\n    pass\n"
    prompt = build_analysis_prompt(patterns, source, file_path="train.py")
    assert "BUG CATALOG" in prompt
    assert "FILE TO ANALYZE: train.py" in prompt
    assert "1 | def train():" in prompt


def test_semantic_analyzer_valid_grounded_finding():
    code = """import torch
features = features[torch.randperm(len(features))]
labels = labels[torch.randperm(len(labels))]
"""
    canned = [
        {
            "bug_id": "label_data_misalignment",
            "line": 3,
            "evidence": "labels = labels[torch.randperm(len(labels))]",
            "message": "Features and labels are shuffled independently with separate permutations",
            "severity": "critical",
            "confidence": 0.95,
        }
    ]
    client = FakeLLMClient(canned_findings=canned)
    analyzer = SemanticAnalyzer(client=client)

    report = analyzer.analyze_code(code, file_path="train.py")
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.bug_id == "label_data_misalignment"
    assert f.detector == "semantic"
    assert f.line == 3
    assert f.confidence == 0.95
    assert f.extra["grounding"] == "verified"
    assert "Verify the same permutation/seed" in f.suggested_fix


def test_semantic_analyzer_ungrounded_evidence_downgrades_confidence():
    code = """import torch
loss = criterion(output, target)
"""
    # Evidence refers to something that doesn't exist near line 2
    canned = [
        {
            "bug_id": "label_data_misalignment",
            "line": 2,
            "evidence": "completely hallucinated line that is nowhere in source",
            "message": "Hallucinated issue",
            "severity": "warning",
            "confidence": 0.90,
        }
    ]
    client = FakeLLMClient(canned_findings=canned)
    analyzer = SemanticAnalyzer(client=client)

    report = analyzer.analyze_code(code, file_path="train.py")
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.extra["grounding"] == "ungrounded_evidence"
    assert f.confidence <= 0.45


def test_semantic_analyzer_drops_unknown_bug_id():
    code = "x = 1\n"
    canned = [
        {
            "bug_id": "completely_invented_bug_id_12345",
            "line": 1,
            "evidence": "x = 1",
            "message": "Invented bug",
            "severity": "warning",
            "confidence": 0.90,
        }
    ]
    client = FakeLLMClient(canned_findings=canned)
    analyzer = SemanticAnalyzer(client=client)

    report = analyzer.analyze_code(code)
    assert len(report.findings) == 0
    assert any("unrecognized bug_id" in d for d in report.diagnostics)


def test_semantic_analyzer_handles_markdown_json_response():
    code = "import torch\nmodel = Net()\n"
    markdown_response = """
Here are the findings:
```json
[
  {
    "bug_id": "data_leakage_train_test",
    "line": 1,
    "evidence": "import torch",
    "message": "Possible data leakage",
    "severity": "warning",
    "confidence": 0.8
  }
]
```
Hope this helps!
"""
    client = FakeLLMClient(responses=[markdown_response])
    analyzer = SemanticAnalyzer(client=client)

    report = analyzer.analyze_code(code)
    assert len(report.findings) == 1
    assert report.findings[0].bug_id == "data_leakage_train_test"


def test_semantic_analyzer_malformed_json_diagnostic():
    code = "import torch\n"
    client = FakeLLMClient(responses=["Not a valid JSON array at all"])
    analyzer = SemanticAnalyzer(client=client)

    report = analyzer.analyze_code(code)
    assert len(report.findings) == 0
    assert any("Failed to parse valid JSON array" in d for d in report.diagnostics)


def test_very_long_files_are_cut_to_fit_one_request():
    from silent_bug_agent.semantic.analyzer import MAX_REVIEW_CHARS

    client = FakeLLMClient(responses=["[]"])
    source = "\n".join(f"x_{i} = {i}" for i in range(5000))
    report = SemanticAnalyzer(client=client).analyze_code(source, "big.py")
    assert len(client.call_history[0]["prompt"]) < MAX_REVIEW_CHARS + 12000
    assert any("too long" in d for d in report.diagnostics)
