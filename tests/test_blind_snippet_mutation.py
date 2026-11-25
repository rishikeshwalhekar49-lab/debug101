"""Blind benchmark copies, snippet mode and the mutation operators: all static, so these tests are fast."""

import ast
import io
import json
from pathlib import Path

import pytest

from silent_bug_agent.cli import main
from silent_bug_agent.mutation.blind import blind_copy, expected_bug, hints_left
from silent_bug_agent.mutation.operators import OPERATORS, mutants
from silent_bug_agent.mutation.templates import Spec, random_spec, render
from silent_bug_agent.static.snippet import NOT_VALIDATED, analyze_snippet, parse_snippet

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
BENCHMARKS = sorted((EXAMPLES / "sample_scripts").glob("benchmark_[0-9]*.py")) + sorted((EXAMPLES / "new").glob("benchmark_[0-9]*.py"))


# ------------------------------------------------------------------- blind
@pytest.mark.parametrize("path", BENCHMARKS, ids=lambda p: p.stem[:12])
def test_blind_copies_leave_no_hint(path):
    source = path.read_text(encoding="utf-8")
    assert expected_bug(source)  # read from the docstring before it is removed
    for fixed in (False, True):
        copy = blind_copy(source, fixed=fixed)
        ast.parse(copy)
        assert hints_left(copy) == [], (path.name, fixed)


def test_only_benchmark_09_has_no_healthy_twin():
    same = [p.name for p in BENCHMARKS if blind_copy(p.read_text()) == blind_copy(p.read_text(), fixed=True)]
    assert same == ["benchmark_09_lr_scheduler_step_order.py"]  # its --fix flag changes nothing


def test_the_fix_switch_is_folded_away():
    source = '''"""Bug: x. Taxonomy ID: demo"""
import argparse
p = argparse.ArgumentParser(description="Benchmark: demo")
p.add_argument("--fix", action="store_true")
args = p.parse_args()
if args.fix:
    good()  # BUG: hint
else:
    bad()
value = 1 if not args.fix else 2
flag = args.fix and ready()
other = args.fix or ready()
'''
    buggy, healthy = blind_copy(source), blind_copy(source, fixed=True)
    assert "bad()" in buggy and "good()" not in buggy and "value = 1" in buggy
    assert "flag = False" in buggy and "other = ready()" in buggy
    assert "good()" in healthy and "value = 2" in healthy and "flag = ready()" in healthy and "other = True" in healthy
    assert "--fix" not in buggy and "description" not in buggy and "#" not in buggy


# ----------------------------------------------------------------- snippet
def test_snippets_are_parsed_leniently_and_lines_point_at_the_paste():
    pasted = "```python\n\nlogits = torch.softmax(model(x), dim=1)\nloss = nn.CrossEntropyLoss()(logits, y)\n```"
    report = analyze_snippet(pasted)
    assert [(f.bug_id, f.line) for f in report.findings] == [("softmax_before_cross_entropy", 3)]
    assert report.diagnostics[0] == NOT_VALIDATED and "removed markdown code fences" in report.diagnostics[-1]
    assert all(f.extra["validated"] is False for f in report.findings)


def test_an_indented_loop_body_is_checked_without_seed_noise():
    pasted = "\n        for x, y in loader:\n            loss = criterion(model(x), y)\n            loss.backward()\n" \
             "            optimizer.step()\n"
    report = analyze_snippet(pasted)
    assert [(f.bug_id, f.line) for f in report.findings] == [("missing_zero_grad", 4)]  # no unset_random_seed


def test_a_method_body_and_a_cut_off_line_still_parse():
    parsed = parse_snippet("    def forward(self, x):\n        return self.net(x)\nloss_fn = nn.MSELoss(\n")
    assert parsed is not None and any("dropped 1" in r for r in parsed.repairs)
    repl = parse_snippet(">>> opt.zero_grad()\n>>> loss.backward()\n")
    assert repl is not None and "removed >>> prompts" in repl.repairs


def test_unparseable_snippets_say_so():
    report = analyze_snippet("for (((")
    assert not report.findings and "could not be parsed" in report.diagnostics[0]


def test_the_snippet_command_reads_standard_input(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("for xb, yb in dl:\n    loss = crit(model(xb), yb)\n    loss.backward()\n"))
    assert main(["snippet", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [f["bug_id"] for f in report["findings"]] == ["missing_zero_grad"]
    assert main(["snippet", str(EXAMPLES / "snippets" / "training_step.txt")]) == 1
    out = capsys.readouterr().out
    assert "softmax_before_cross_entropy" in out and "missing_zero_grad" in out and "not validated" in out


# ---------------------------------------------------------------- mutation
def test_a_template_with_dropout_and_relu_gets_every_bug():
    spec = Spec(style="class", dropout=0.1, activation="relu", optimizer="sgd", lr=0.1)
    found = mutants(render(spec))
    assert set(found) == set(OPERATORS)
    for code in found.values():
        ast.parse(code)
    assert "zero_grad" not in found["missing_zero_grad"]
    assert ".eval()" not in found["train_eval_mode_mismatch"]
    assert ("torch.softmax(self.out(x), dim=1)" in found["softmax_before_cross_entropy"]
            or "criterion(torch.softmax(output, dim=1), yb)" in found["softmax_before_cross_entropy"])
    assert "reduction='sum'" in found["wrong_loss_reduction"]
    assert "lr=10.0" in found["learning_rate_misconfigured"]
    assert ("requires_grad_(False)" in found["frozen_layer_no_grad"]
            or "list(model.parameters())[2:]" in found["frozen_layer_no_grad"])
    assert "-8.0" in found["dead_relu"]
    assert "* 255.0 + 128.0" in found["non_normalized_inputs"]
    assert "y_train = y_train[torch.randperm(len(y_train))]" in found["label_data_misalignment"]


def test_bugs_that_cannot_happen_are_not_planted():
    regression = render(Spec(task="regression", activation="tanh", dropout=0.0, batchnorm=False, style="sequential"))
    found = mutants(regression)
    assert "softmax_before_cross_entropy" not in found  # no CrossEntropyLoss
    assert "dead_relu" not in found  # no ReLU
    assert "train_eval_mode_mismatch" not in found  # eval() changes nothing without dropout or batch norm
    sequential = mutants(render(Spec(style="sequential")))["softmax_before_cross_entropy"]
    assert "nn.Softmax(dim=1)" in sequential or "criterion(torch.softmax(output, dim=1), yb)" in sequential
    adam = mutants(render(Spec(optimizer="adam", lr=0.002)))["learning_rate_misconfigured"]
    assert "lr=2.0" in adam or "lr=2e-07" in adam  # adaptive: a larger factor, or (one script in three) far too small


def test_random_templates_always_render_valid_code():
    import random

    rng = random.Random(3)
    for i in range(40):
        ast.parse(render(random_spec(rng, i)))


# ------------------------------------------------------- hand-written test set
def test_the_hand_written_test_set_applies_cleanly_and_gives_nothing_away():
    import importlib.util

    spec = importlib.util.spec_from_file_location("holdout_variants", EXAMPLES / "holdout" / "variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = module.versions()  # raises if any edit does not match exactly once
    assert len(versions) == 97 and sum(v["split"] == "test" for v in versions) == 58
    assert {v["label"] for v in versions} == set(OPERATORS) | {"clean"}
    for v in versions:
        ast.parse(v["code"])
        assert [h for h in hints_left(v["code"]) if h != "the word 'fix'"] == [], (v["script"], v["label"])
    codes = [v["code"] for v in versions]
    assert len(set(codes)) == len(codes)  # every version differs from the others


def test_each_bug_can_be_planted_more_than_one_way():
    import random

    rng = random.Random(0)
    frozen, softmax = set(), set()
    for i in range(40):
        found = mutants(render(random_spec(rng, i)))
        if "frozen_layer_no_grad" in found:
            frozen.add("left out" if "parameters())[2:]" in found["frozen_layer_no_grad"] else "requires_grad")
        if "softmax_before_cross_entropy" in found:
            softmax.add("in loss" if "criterion(torch.softmax" in found["softmax_before_cross_entropy"] else "in model")
    assert frozen == {"left out", "requires_grad"} and softmax == {"in loss", "in model"}
