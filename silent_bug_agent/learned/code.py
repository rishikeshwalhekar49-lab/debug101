"""Features read from the script's code, next to the ones measured from its run (version 4).

A trace says what happened; the code often says why. These are plain facts about the
source, read from its syntax tree, never a verdict: whether every `backward()` call
sits in a loop that also clears gradients (in any of the usual ways), whether the
model has dropout or batch norm and the script ever calls `eval()`, whether a softmax
and a cross-entropy loss both appear, whether a loss sums instead of averaging,
whether some parameters stop requiring gradients, whether the inputs are scaled,
whether labels are permuted on their own, and the learning rate written in the code.

A script that does not parse gives zeros and `code_parsed = 0`. The classifier learns
how far to trust each fact: a healthy script can sum its loss on purpose (with a
smaller learning rate), and a softmax can be there only to print probabilities.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, List, Optional

CODE_FEATURES: List[str] = [
    "code_parsed",
    "code_backward_covered",  # share of backward() calls with gradient clearing in the same loop (or function)
    "code_mode_layers",  # Dropout or BatchNorm appears
    "code_has_eval",  # .eval() or .train(False) is called
    "code_softmax_and_ce",  # a (non-log) softmax and a cross-entropy loss both appear
    "code_reduction_sum",  # a loss built with reduction="sum" (or size_average=False)
    "code_sum_backward",  # backward() called on a .sum(): usually paired with dividing the step by the batch size
    "code_requires_grad_false",
    "code_optimizer_partial",  # an optimizer is given something other than model.parameters()
    "code_scales_inputs",  # / 255, (x - mean) / std, Normalize(...), a scaler
    "code_labels_permuted",  # labels indexed by a permutation that no input uses
    "code_log_lr",  # log10 of the learning rate written in the code
    "code_has_lr",
]
_LABEL = re.compile(r"^(y|Y|labels?|targets?|tags?)(_?(train|tr|batch))?$|_(labels?|targets?|y)$")
_LR_NAMES = {"lr", "learning_rate", "LR", "LEARNING_RATE", "base_lr", "step_size"}
_OPTIMIZERS = {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adamax", "NAdam", "RAdam", "Adadelta"}


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _chain(node: ast.AST) -> List[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _clears_gradients(node: ast.AST, helpers: frozenset = frozenset()) -> bool:
    """Gradient clearing inside `node`, or a call to one of `helpers` (functions of this file that clear them)."""
    for sub in ast.walk(node):
        if helpers and isinstance(sub, ast.Call) and _name(sub) in helpers:
            return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == "zero_grad":
                return True
            if sub.func.attr in ("zero_", "detach_", "fill_") and "grad" in _chain(sub.func.value):
                return True
        if isinstance(sub, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "grad" for t in sub.targets):
            return True
    return False


def _parents(tree: ast.AST) -> Dict[int, ast.AST]:
    parents: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _backward_covered(tree: ast.AST) -> Optional[float]:
    parents = _parents(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n) == "backward"]
    if not calls:
        return None
    helpers = frozenset(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and _clears_gradients(n))
    covered = 0
    for call in calls:
        node, scope = call, None
        while id(node) in parents:
            node = parents[id(node)]
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                scope = node
                break
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = node
                break
        covered += int(scope is not None and _clears_gradients(scope, helpers))
    return covered / len(calls)


def _labels_permuted(tree: ast.AST) -> bool:
    permutations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and _name(node.value) in ("randperm", "permutation"):
            permutations |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    indexes: Dict[str, set] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            target = node.value.id
            index = node.slice
            if isinstance(index, ast.Call) and _name(index) in ("randperm", "permutation") and _LABEL.search(target):
                return True
            if isinstance(index, ast.Name) and index.id in permutations:
                indexes.setdefault(index.id, set()).add(target)
        if isinstance(node, ast.Call) and _name(node) == "shuffle" and node.args \
                and isinstance(node.args[0], ast.Name) and _LABEL.search(node.args[0].id):
            return True
        if isinstance(node, ast.Call) and _name(node) == "roll" and node.args \
                and isinstance(node.args[0], ast.Name) and _LABEL.search(node.args[0].id):
            return True
    return any(any(_LABEL.search(t) for t in targets) and all(_LABEL.search(t) for t in targets)
               for targets in indexes.values())


def _scales_inputs(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)) and abs(right.value) >= 100:
                return True
            if "std" in " ".join(_chain(right)).lower() or (isinstance(right, ast.Call) and _name(right) == "std"):
                return True
        if isinstance(node, ast.Call) and _name(node) in ("Normalize", "normalize", "StandardScaler", "MinMaxScaler",
                                                           "scale"):
            return True
    return False


def _learning_rate(tree: ast.AST) -> Optional[float]:
    constants: Dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    constants[t.id] = float(node.value.value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):  # lr, epochs = 0.1, 10
            for t in node.targets:
                if isinstance(t, ast.Tuple) and len(t.elts) == len(node.value.elts):
                    for name, value in zip(t.elts, node.value.elts):
                        if isinstance(name, ast.Name) and isinstance(value, ast.Constant) \
                                and isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                            constants[name.id] = float(value.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node) in _OPTIMIZERS:
            lr = next((k.value for k in node.keywords if k.arg == "lr"), node.args[1] if len(node.args) > 1 else None)
            if isinstance(lr, ast.Constant) and isinstance(lr.value, (int, float)):
                return float(lr.value)
            if isinstance(lr, ast.Name) and lr.id in constants:
                return constants[lr.id]
            if lr is None and _name(node) == "SGD":
                return None
            if lr is None:
                return 1e-3  # the default of Adam and friends
    for name in _LR_NAMES:
        if name in constants:
            return constants[name]
    return None


def code_features(source: str) -> Dict[str, float]:
    zeros = {name: 0.0 for name in CODE_FEATURES}
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, ValueError):
        return zeros
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = {_name(n) for n in calls} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
        | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    covered = _backward_covered(tree)
    softmax = any(_name(c) in ("softmax", "Softmax") for c in calls)
    cross_entropy = bool(names & {"CrossEntropyLoss", "cross_entropy"})
    reduction_sum = any(any((k.arg == "reduction" and isinstance(k.value, ast.Constant) and k.value.value == "sum")
                            or (k.arg == "size_average" and isinstance(k.value, ast.Constant) and k.value.value is False)
                            for k in c.keywords) for c in calls)
    sum_backward = any(_name(c) == "backward" and isinstance(c.func, ast.Attribute)
                       and isinstance(c.func.value, ast.Call) and _name(c.func.value) == "sum" for c in calls)
    requires_grad_false = any(
        (_name(c) == "requires_grad_" and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value is False)
        or any(k.arg == "requires_grad" and isinstance(k.value, ast.Constant) and k.value.value is False
               for k in c.keywords) for c in calls) or any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "requires_grad" for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value is False for n in ast.walk(tree))
    optimizers = [c for c in calls if _name(c) in _OPTIMIZERS and c.args]
    partial = any(not (isinstance(c.args[0], ast.Call) and _name(c.args[0]) == "parameters"
                       and isinstance(c.args[0].func, ast.Attribute) and isinstance(c.args[0].func.value, ast.Name))
                  for c in optimizers)
    has_eval = any(_name(c) == "eval" and isinstance(c.func, ast.Attribute) and not c.args for c in calls) or any(
        _name(c) == "train" and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value is False
        for c in calls)
    lr = _learning_rate(tree)
    return {
        "code_parsed": 1.0,
        "code_backward_covered": covered if covered is not None else 0.0,
        "code_mode_layers": 1.0 if any("Dropout" in n or "BatchNorm" in n or n == "dropout" for n in names) else 0.0,
        "code_has_eval": 1.0 if has_eval else 0.0,
        "code_softmax_and_ce": 1.0 if softmax and cross_entropy else 0.0,
        "code_reduction_sum": 1.0 if reduction_sum else 0.0,
        "code_sum_backward": 1.0 if sum_backward else 0.0,
        "code_requires_grad_false": 1.0 if requires_grad_false else 0.0,
        "code_optimizer_partial": 1.0 if partial else 0.0,
        "code_scales_inputs": 1.0 if _scales_inputs(tree) else 0.0,
        "code_labels_permuted": 1.0 if _labels_permuted(tree) else 0.0,
        "code_log_lr": math.log10(lr) if lr and lr > 0 else 0.0,
        "code_has_lr": 1.0 if lr and lr > 0 else 0.0,
    }
