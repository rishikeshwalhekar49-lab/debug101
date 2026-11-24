"""Mutation operators: each plants one known silent bug in a healthy PyTorch script.

An operator reads the script's syntax tree, finds what it needs by the usual
names (an optimizer built from `X.parameters()`, a loss built from
`nn.CrossEntropyLoss()`, inputs called `x...`, labels called `y...`), and returns
the broken script, or None when the bug does not apply (no ReLU to kill, no
dropout or batch norm for `eval()` to matter, ...). The operators work on the
generated templates and on ordinary scripts written the same way.

Every operator's name is the taxonomy id of the bug it plants, so a mutant's
label is exact.
"""

from __future__ import annotations

import ast
import re
from typing import Callable, Dict, List, Optional

LOSSES = {"CrossEntropyLoss", "MSELoss", "L1Loss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss", "SmoothL1Loss",
          "HuberLoss"}
OPTIMIZER_NAMES = {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adamax", "NAdam", "RAdam"}
LR_FACTOR = 100.0  # SGD
ADAPTIVE_LR_FACTOR = 1000.0  # Adam and friends shrug off x100 on easy problems, so it would not be a bug there
ADAPTIVE = {"Adam", "AdamW", "RMSprop", "Adagrad", "Adamax", "NAdam", "RAdam"}
DEAD_BIAS = -8.0
_INPUT_NAME = re.compile(r"^(x|X|inputs?|features?|data)(_?(train|val|valid|test))?$")
_LABEL_NAME = re.compile(r"^(y|Y|labels?|targets?)(_?train)?$")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _statement(code: str) -> List[ast.stmt]:
    return ast.parse(code).body


def _bodies(tree: ast.AST):
    for node in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if isinstance(body, list) and body and all(isinstance(s, ast.stmt) for s in body):
                yield body


def _finish(tree: ast.Module) -> str:
    for body in _bodies(tree):
        if not body:
            body.append(ast.Pass())
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _remove_method_calls(source: str, method: str) -> Optional[str]:
    tree = ast.parse(source)
    removed = 0
    for body in _bodies(tree):
        kept = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                                         and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == method)]
        removed += len(body) - len(kept)
        body[:] = kept or [ast.Pass()]
    return _finish(tree) if removed else None


def _trained_model(tree: ast.Module) -> Optional[str]:
    """The name X in `optim.Something(X.parameters(), ...)`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in OPTIMIZER_NAMES and node.args:
            first = node.args[0]
            if isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute) \
                    and first.func.attr == "parameters" and isinstance(first.func.value, ast.Name):
                return first.func.value.id
    return None


def _insert_after_assignment(tree: ast.Module, name: str, code: str) -> bool:
    for body in _bodies(tree):
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                         for t in statement.targets):
                body[index + 1:index + 1] = _statement(code)
                return True
    return False


# --------------------------------------------------------------- operators
def missing_zero_grad(source: str) -> Optional[str]:
    return _remove_method_calls(source, "zero_grad")


def train_eval_mode_mismatch(source: str) -> Optional[str]:
    if "Dropout" not in source and "BatchNorm" not in source:
        return None  # without them eval() changes nothing, so leaving it out is not a bug
    return _remove_method_calls(source, "eval")


def _softmax_in_model(tree: ast.Module, model: str) -> bool:
    """Make the model itself return probabilities, the way this bug is usually written."""
    built = next((node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == model for t in node.targets)), None)
    if not isinstance(built, ast.Call):
        return False
    if _call_name(built) == "Sequential":
        built.args.append(ast.parse("nn.Softmax(dim=1)", mode="eval").body)
        return True
    cls = next((node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                and node.name == _call_name(built)), None)
    forward = next((node for node in (cls.body if cls else []) if isinstance(node, ast.FunctionDef)
                    and node.name == "forward"), None)
    returns = [node for node in ast.walk(forward) if isinstance(node, ast.Return) and node.value] if forward else []
    for node in returns:
        node.value = ast.parse(f"torch.softmax({ast.unparse(node.value)}, dim=1)", mode="eval").body
    return bool(returns)


def softmax_before_cross_entropy(source: str) -> Optional[str]:
    tree = ast.parse(source)
    criteria = {t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and _call_name(node.value) == "CrossEntropyLoss" for t in node.targets if isinstance(t, ast.Name)}
    if not criteria:
        return None
    model = _trained_model(tree)
    if model and _softmax_in_model(tree, model):
        return _finish(tree)
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in criteria and node.args:
            node.args[0] = ast.parse(f"torch.softmax({ast.unparse(node.args[0])}, dim=1)", mode="eval").body
            changed += 1
    return _finish(tree) if changed else None


def wrong_loss_reduction(source: str) -> Optional[str]:
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in LOSSES:
            current = next((k for k in node.keywords if k.arg == "reduction"), None)
            if current is not None and isinstance(current.value, ast.Constant) and current.value.value == "sum":
                continue
            node.keywords = [k for k in node.keywords if k.arg != "reduction"]
            node.keywords.append(ast.keyword(arg="reduction", value=ast.Constant("sum")))
            changed += 1
    return _finish(tree) if changed else None


def learning_rate_misconfigured(source: str) -> Optional[str]:
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in OPTIMIZER_NAMES:
            factor = ADAPTIVE_LR_FACTOR if _call_name(node) in ADAPTIVE else LR_FACTOR
            lr = next((k for k in node.keywords if k.arg == "lr"), None)
            if lr is None:
                node.keywords.append(ast.keyword(arg="lr", value=ast.Constant(1.0)))
            elif isinstance(lr.value, ast.Constant) and isinstance(lr.value.value, (int, float)):
                lr.value = ast.Constant(round(float(lr.value.value) * factor, 6))
            else:
                lr.value = ast.BinOp(left=lr.value, op=ast.Mult(), right=ast.Constant(factor))
            changed += 1
    return _finish(tree) if changed else None


def frozen_layer_no_grad(source: str) -> Optional[str]:
    tree = ast.parse(source)
    model = _trained_model(tree)
    code = f"for _p in list({model}.parameters())[:2]:\n    _p.requires_grad_(False)"
    return _finish(tree) if model and _insert_after_assignment(tree, model, code) else None


def dead_relu(source: str) -> Optional[str]:
    if not re.search(r"\brelu\b|ReLU\(", source):
        return None
    tree = ast.parse(source)
    model = _trained_model(tree)
    code = (f"for _m in {model}.modules():\n"
            "    if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d)) and _m.bias is not None:\n"
            f"        torch.nn.init.constant_(_m.bias, {DEAD_BIAS})")
    return _finish(tree) if model and _insert_after_assignment(tree, model, code) else None


def non_normalized_inputs(source: str) -> Optional[str]:
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and _INPUT_NAME.match(node.targets[0].id) and isinstance(node.value, ast.Call):
            node.value = ast.parse(f"({ast.unparse(node.value)}) * 255.0 + 128.0", mode="eval").body
            changed += 1
    return _finish(tree) if changed else None


def label_data_misalignment(source: str) -> Optional[str]:
    tree = ast.parse(source)
    names = [t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
             for t in node.targets if isinstance(t, ast.Name) and _LABEL_NAME.match(t.id)]
    if not names:
        return None
    name = next((n for n in names if "train" in n), names[0])
    code = f"{name} = {name}[torch.randperm(len({name}))]"
    return _finish(tree) if _insert_after_assignment(tree, name, code) else None


OPERATORS: Dict[str, Callable[[str], Optional[str]]] = {
    "missing_zero_grad": missing_zero_grad,
    "train_eval_mode_mismatch": train_eval_mode_mismatch,
    "softmax_before_cross_entropy": softmax_before_cross_entropy,
    "wrong_loss_reduction": wrong_loss_reduction,
    "learning_rate_misconfigured": learning_rate_misconfigured,
    "frozen_layer_no_grad": frozen_layer_no_grad,
    "dead_relu": dead_relu,
    "non_normalized_inputs": non_normalized_inputs,
    "label_data_misalignment": label_data_misalignment,
}


def mutants(source: str) -> Dict[str, str]:
    """Every bug that applies to this script, as {bug id: broken script}."""
    out = {}
    for bug, operator in OPERATORS.items():
        try:
            broken = operator(source)
        except SyntaxError:
            return {}
        if broken is not None and broken != source:
            out[bug] = broken
    return out
