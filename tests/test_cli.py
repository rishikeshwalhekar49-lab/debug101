"""Tests for the CLI."""

import json
from pathlib import Path
import pytest

from silent_bug_agent.cli import build_parser, main, run_scan


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["scan", "train.py"])
    assert args.command == "scan"
    assert args.path == "train.py"
    assert not args.semantic
    assert not args.json


def test_cli_parser_flags():
    parser = build_parser()
    args = parser.parse_args(["scan", "train.py", "--semantic", "--json", "--model", "custom-model"])
    assert args.semantic
    assert args.json
    assert args.model == "custom-model"


def test_cli_scan_nonexistent_path(capsys):
    ret = run_scan("non_existent_file_path_1234.py")
    assert ret == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cli_scan_clean_file(tmp_path, capsys):
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("import torch\nprint('hello')\n")

    ret = run_scan(str(clean_file), json_output=False)
    assert ret == 0
    captured = capsys.readouterr()
    assert "0 finding(s)" in captured.out


def test_cli_scan_buggy_file_json(tmp_path, capsys):
    buggy_file = tmp_path / "buggy.py"
    buggy_file.write_text("""
for data, target in loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
""")

    ret = run_scan(str(buggy_file), json_output=True)
    assert ret == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["summary"]["total_findings"] >= 1
    bug_ids = [f["bug_id"] for f in data["findings"]]
    assert "missing_zero_grad" in bug_ids


def test_cli_main_entry(tmp_path):
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("x = 1\n")
    ret = main(["scan", str(clean_file)])
    assert ret == 0
