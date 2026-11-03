"""Pure-Python patching, the safety guard, and every repair recipe on the benchmarks."""

from pathlib import Path

import pytest

from silent_bug_agent.repair.patch_utils import (
    MalformedDiffError,
    apply_diff_in_temp,
    apply_patch_to_file,
    apply_unified_diff,
    generate_minimal_diff,
    patch_guard,
)
from silent_bug_agent.repair.recipes import recipe_for

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_scripts"


@pytest.mark.parametrize("original,updated", [
    ("a\nb\nc\n", "a\nB\nc\n"),
    ("a\nb\nc", "a\nb\nc\nd"),  # no newline at the end
    ("x = 1\n", "import torch\nx = 1\n"),
])
def test_diff_round_trip(original, updated):
    assert apply_unified_diff(original, generate_minimal_diff(original, updated, "t.py")) == updated


def test_mismatched_context_is_refused():
    diff = generate_minimal_diff("a\nb\n", "a\nc\n", "t.py")
    with pytest.raises(MalformedDiffError):
        apply_unified_diff("z\nb\n", diff)


def test_apply_in_temp_needs_no_git(tmp_path):
    target = tmp_path / "train.py"
    target.write_text("lr = 0.1\n", encoding="utf-8")
    ok, message, updated = apply_diff_in_temp(str(target), generate_minimal_diff("lr = 0.1\n", "lr = 0.01\n", "train.py"))
    assert ok and updated == "lr = 0.01\n"
    assert target.read_text(encoding="utf-8") == "lr = 0.1\n"


def test_apply_to_file_keeps_windows_line_endings_and_a_backup(tmp_path):
    target = tmp_path / "train.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\n")
    diff = generate_minimal_diff("a = 1\nb = 2\n", "a = 1\nb = 3\n", "train.py")
    backup = apply_patch_to_file(str(target), diff)
    assert target.read_bytes() == b"a = 1\r\nb = 3\r\n"
    assert backup.read_bytes() == b"a = 1\r\nb = 2\r\n"


@pytest.mark.parametrize("added", ["import subprocess", "os.system('rm -rf /')", "eval('1')", "exec(code)", "sys.exit(0)"])
def test_guard_refuses_dangerous_additions(added):
    assert patch_guard("x = 1\n", f"x = 1\n{added}\n") is not None


def test_guard_allows_ordinary_training_calls():
    assert patch_guard("loss.backward()\nopt.step()\n", "model.eval()\nloss.backward()\nopt.step()\n") is None


def test_guard_refuses_removing_the_training_step():
    assert "step" in patch_guard("loss.backward()\nopt.step()\n", "loss.backward()\n")


EXPECTED = {
    "benchmark_01_missing_zero_grad.py": ("missing_zero_grad", "optimizer.zero_grad()"),
    "benchmark_02_train_eval_mode.py": ("train_eval_mode_mismatch", "model.eval()"),
    "benchmark_03_frozen_layer.py": ("frozen_layer_no_grad", "optim.SGD(model.parameters(), lr=0.05)"),
    "benchmark_04_softmax_before_crossentropy.py": ("softmax_before_cross_entropy", "return out"),
    "benchmark_05_learning_rate_too_high.py": ("learning_rate_misconfigured", "lr = 0.01 if args.fix else 0.01"),
    "benchmark_06_dead_relu.py": ("dead_relu", "self.fc1.bias.fill_(0.0)"),
    "benchmark_07_wrong_loss_reduction.py": ("wrong_loss_reduction", 'reduction = "mean" if args.fix else "mean"'),
    "benchmark_08_unnormalized_inputs.py": ("non_normalized_inputs", "x = (x - x.mean(dim=0, keepdim=True))"),
}


@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_recipe_repairs_each_benchmark(script):
    bug_id, expected = EXPECTED[script]
    source = (SAMPLES / script).read_text(encoding="utf-8")
    recipe = recipe_for(bug_id, source)
    assert recipe is not None, bug_id
    assert expected in recipe.source
    compile(recipe.source, script, "exec")
    assert patch_guard(source, recipe.source) is None


def test_exploding_gradients_fixes_the_learning_rate_when_it_is_huge():
    source = (SAMPLES / "benchmark_05_learning_rate_too_high.py").read_text(encoding="utf-8")
    assert "0.01 if args.fix else 0.01" in recipe_for("exploding_gradients", source).source


def test_recipes_leave_clean_code_alone():
    clean = ("import torch\nmodel = torch.nn.Linear(2, 1)\nopt = torch.optim.SGD(model.parameters(), lr=0.01)\n"
             "for x in range(3):\n    opt.zero_grad()\n    loss = model(torch.ones(2)).sum()\n    loss.backward()\n"
             "    opt.step()\n")
    assert recipe_for("missing_zero_grad", clean) is None
    assert recipe_for("softmax_before_cross_entropy", clean) is None
    assert recipe_for("frozen_layer_no_grad", clean) is None
    assert recipe_for("no_such_bug", clean) is None


def test_a_second_fix_keeps_the_true_original_as_backup(tmp_path):
    target = tmp_path / "train.py"
    target.write_text("a = 1\nb = 1\n", encoding="utf-8")
    apply_patch_to_file(str(target), generate_minimal_diff("a = 1\nb = 1\n", "a = 2\nb = 1\n", "train.py"))
    backup = apply_patch_to_file(str(target), generate_minimal_diff("a = 2\nb = 1\n", "a = 2\nb = 2\n", "train.py"))
    assert target.read_text(encoding="utf-8") == "a = 2\nb = 2\n"
    assert backup.read_text(encoding="utf-8") == "a = 1\nb = 1\n"
