"""Rule-based repairs for known bugs: the offline path of the Fix Synthesizer.

Each recipe rewrites the source using its syntax tree and returns the new text,
or None when the pattern is not there. An LLM handles what these cannot.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..schemas import RepairStrategy

_LOOPS = (ast.For, ast.While)
_OPTIMIZERS = {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Adamax", "NAdam", "RAdam", "ASGD", "LBFGS"}


@dataclass
class Recipe:
    source: str
    note: str
    strategy: RepairStrategy = RepairStrategy.DIRECT_EDIT


def _dotted(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _call_name(node: ast.Call) -> str:
    return _dotted(node.func) or ""


def _calls(root: ast.AST) -> Iterable[ast.Call]:
    return (n for n in ast.walk(root) if isinstance(n, ast.Call))


def _has_method(root: ast.AST, method: str) -> bool:
    return any(isinstance(c.func, ast.Attribute) and c.func.attr == method for c in _calls(root))


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _insert(source: str, lineno: int, column: int, statements: List[str]) -> str:
    lines = source.split("\n")
    lines[lineno - 1:lineno - 1] = [" " * column + s for s in statements]
    return "\n".join(lines)


def _replace(source: str, node: ast.AST, text: str) -> str:
    lines = source.split("\n")
    first, last = node.lineno - 1, node.end_lineno - 1
    head = lines[first].encode("utf-8")[:node.col_offset].decode("utf-8")
    tail = lines[last].encode("utf-8")[node.end_col_offset:].decode("utf-8")
    lines[first:last + 1] = [head + text + tail]
    return "\n".join(lines)


def _span(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def _training_loop(tree: ast.AST) -> Optional[ast.AST]:
    """The innermost loop that calls backward()."""
    loops = [n for n in ast.walk(tree) if isinstance(n, _LOOPS) and _has_method(n, "backward")]
    return min(loops, key=lambda n: n.end_lineno - n.lineno) if loops else None


def _optimizer_name(tree: ast.AST, loop: Optional[ast.AST]) -> Optional[str]:
    receivers = [_dotted(c.func.value) for c in _calls(loop or tree)
                 if isinstance(c.func, ast.Attribute) and c.func.attr == "step"]
    names = [r for r in receivers if r and "sched" not in r.lower()]
    return next((r for r in names if "opt" in r.lower()), names[0] if names else None)


def _optimizer_calls(tree: ast.AST) -> List[ast.Call]:
    return [c for c in _calls(tree) if _call_name(c).split(".")[-1] in _OPTIMIZERS]


def _model_names(tree: ast.AST) -> List[str]:
    seen: List[str] = []
    for call in _calls(tree):
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("parameters", "named_parameters", "eval", "train"):
            root = call.func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id not in seen:
                seen.append(root.id)
    return seen


def _assignment(tree: ast.AST, name: str) -> Optional[ast.Assign]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    return None


def missing_zero_grad(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    loop = _training_loop(tree)
    optimizer = _optimizer_name(tree, loop)
    if loop is None or optimizer is None:
        return None
    for statement in loop.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) \
                and _call_name(statement.value).endswith("zero_grad"):
            return None
    first = loop.body[0]
    return Recipe(_insert(source, first.lineno, first.col_offset, [f"{optimizer}.zero_grad()"]),
                  f"clear gradients with {optimizer}.zero_grad() at the start of each step")


def softmax_before_cross_entropy(source: str) -> Optional[Recipe]:
    if "CrossEntropy" not in source and "cross_entropy" not in source:
        return None
    tree = ast.parse(source)
    forwards = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    for root in forwards or [tree]:
        for call in _calls(root):
            name = _call_name(call)
            if name.split(".")[-1] == "softmax" and name in ("torch.softmax", "F.softmax", "nn.functional.softmax",
                                                             "torch.nn.functional.softmax") and call.args:
                return Recipe(_replace(source, call, _span(source, call.args[0])),
                              "return raw logits, because CrossEntropyLoss applies LogSoftmax itself")
            if isinstance(call.func, ast.Attribute) and call.func.attr == "softmax" and name.split(".")[0] not in ("torch", "F", "nn"):
                return Recipe(_replace(source, call, _span(source, call.func.value)),
                              "return raw logits, because CrossEntropyLoss applies LogSoftmax itself")
    for call in _calls(tree):
        if _call_name(call).split(".")[-1] == "Softmax":
            return Recipe(_replace(source, call, "nn.Identity()"),
                          "remove the Softmax layer, because CrossEntropyLoss applies LogSoftmax itself")
    return None


def train_eval_mode(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    parents = _parents(tree)
    models = _model_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        managers = [_call_name(i.context_expr) for i in node.items if isinstance(i.context_expr, ast.Call)]
        if not any(m.endswith(("no_grad", "inference_mode")) for m in managers):
            continue
        called = [c.func.id for c in _calls(node) if isinstance(c.func, ast.Name) and c.func.id in models]
        if not called:
            continue
        model = called[0]
        owner = parents.get(node)
        body = next((b for b in (getattr(owner, "body", None), getattr(owner, "orelse", None))
                     if isinstance(b, list) and node in b), [])
        before = body[:body.index(node)] if node in body else []
        if any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _call_name(s.value) == f"{model}.eval"
               for s in before):
            continue
        ancestor, in_training = parents.get(node), False
        while ancestor is not None:
            in_training = in_training or (isinstance(ancestor, _LOOPS) and _has_method(ancestor, "backward"))
            ancestor = parents.get(ancestor)
        patched = source
        if in_training:
            patched = _insert(patched, node.end_lineno + 1, node.col_offset, [f"{model}.train()"])
        patched = _insert(patched, node.lineno, node.col_offset, [f"{model}.eval()"])
        return Recipe(patched, f"switch {model} to eval mode for validation")
    return None


def frozen_layer(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    for call in _optimizer_calls(tree):
        if not call.args:
            continue
        argument = call.args[0]
        if isinstance(argument, ast.Call) and _call_name(argument).endswith(".parameters"):
            parts = _call_name(argument).split(".")
            if len(parts) > 2:
                return Recipe(_replace(source, argument, f"{parts[0]}.parameters()"),
                              f"give the optimizer every parameter of {parts[0]}")
        if isinstance(argument, ast.Name):
            assign = _assignment(tree, argument.id)
            if assign is not None and isinstance(assign.value, ast.ListComp):
                roots = [_call_name(c).split(".")[0] for c in _calls(assign.value)
                         if _call_name(c).endswith(("named_parameters", "parameters"))]
                if roots:
                    return Recipe(_replace(source, assign.value, f"list({roots[0]}.parameters())"),
                                  f"give the optimizer every parameter of {roots[0]}")
    return None


def _learning_rate_nodes(tree: ast.AST) -> List[ast.Constant]:
    found: List[ast.Constant] = []
    for call in _optimizer_calls(tree):
        values = [k.value for k in call.keywords if k.arg == "lr"] + call.args[1:2]
        for value in values:
            if isinstance(value, ast.Name):
                assign = _assignment(tree, value.id)
                value = assign.value if assign else value
            found += [n for n in ast.walk(value) if isinstance(n, ast.Constant)
                      and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    return found


def learning_rate(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    nodes = [n for n in _learning_rate_nodes(tree) if n.value > 0]
    if not nodes:
        return None
    node = max(nodes, key=lambda n: n.value)
    value = float(node.value)
    new = 0.01 if value >= 1.0 else value / 10.0
    return Recipe(_replace(source, node, repr(round(new, 8))), f"lower the learning rate from {value:g} to {new:g}",
                  RepairStrategy.HYPERPARAM)


def dead_relu(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    for call in _calls(tree):
        name = _call_name(call)
        target = None
        if name.endswith(".bias.fill_") and call.args:
            target = call.args[0]
        elif name.endswith("init.constant_") and len(call.args) > 1 and ".bias" in _span(source, call.args[0]):
            target = call.args[1]
        if target is not None and isinstance(target, ast.UnaryOp) and isinstance(target.op, ast.USub):
            return Recipe(_replace(source, target, "0.0"), "remove the large negative bias that kills the ReLU units")
    for call in _calls(tree):
        name = _call_name(call)
        if name.split(".")[-1] == "ReLU":
            return Recipe(_replace(source, call, "nn.LeakyReLU(0.01)"), "use LeakyReLU so dead units can recover")
        if name in ("F.relu", "torch.relu") and call.args:
            inner = _span(source, call.args[0])
            return Recipe(_replace(source, call, f"torch.nn.functional.leaky_relu({inner}, 0.01)"),
                          "use leaky_relu so dead units can recover")
    return None


def wrong_loss_reduction(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    for call in _calls(tree):
        for keyword in call.keywords:
            if keyword.arg != "reduction":
                continue
            value = keyword.value
            if isinstance(value, ast.Name):
                assign = _assignment(tree, value.id)
                value = assign.value if assign else value
            for node in ast.walk(value):
                if isinstance(node, ast.Constant) and node.value == "sum":
                    return Recipe(_replace(source, node, '"mean"'), "use reduction='mean' so step size does not grow with batch size")
    return None


def non_normalized_inputs(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    for call in _calls(tree):
        if _call_name(call).split(".")[-1] == "Compose" and call.args and isinstance(call.args[0], ast.List):
            items = call.args[0].elts
            names = [_call_name(i).split(".")[-1] for i in items if isinstance(i, ast.Call)]
            if "ToTensor" in names and "Normalize" not in names:
                prefix = _call_name(items[names.index("ToTensor")]).rsplit(".", 1)[0]
                position = items[names.index("ToTensor")]
                addition = f"{prefix}.Lambda(lambda t: (t - t.mean()) / (t.std() + 1e-7))"
                return Recipe(_replace(source, position, f"{_span(source, position)}, {addition}"),
                              "normalise each input after ToTensor")
    loop = _training_loop(tree)
    if loop is None:
        return None
    models = _model_names(tree)
    for call in _calls(loop):
        if isinstance(call.func, ast.Name) and call.func.id in models and call.args and isinstance(call.args[0], ast.Name):
            name = call.args[0].id
            statement = f"{name} = ({name} - {name}.mean(dim=0, keepdim=True)) / ({name}.std(dim=0, keepdim=True) + 1e-7)"
            inside = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store) for n in ast.walk(loop))
            if inside:
                parents = _parents(tree)
                node: ast.AST = call
                while not isinstance(node, ast.stmt):
                    node = parents[node]
                return Recipe(_insert(source, node.lineno, node.col_offset, [statement]), f"normalise {name} before the model")
            return Recipe(_insert(source, loop.lineno, loop.col_offset, [statement]), f"normalise {name} before training")
    return None


def exploding_gradients(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    largest = max((float(n.value) for n in _learning_rate_nodes(tree)), default=0.0)
    if largest >= 1.0:  # the usual root cause, so fix that and not the symptom
        return learning_rate(source)
    models = _model_names(tree)
    loop = _training_loop(tree)
    if loop is None or not models:
        return None
    for node in ast.walk(loop):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and _call_name(node.value).endswith(".backward"):
            return Recipe(_insert(source, node.end_lineno + 1, node.col_offset,
                                  [f"torch.nn.utils.clip_grad_norm_({models[0]}.parameters(), 1.0)"]),
                          "clip the gradient norm to 1.0")
    return None


def unset_random_seed(source: str) -> Optional[Recipe]:
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports or "torch" not in source:
        return None
    return Recipe(_insert(source, imports[-1].end_lineno + 1, 0, ["", "torch.manual_seed(0)"]), "set a fixed random seed")


RECIPES: Dict[str, Callable[[str], Optional[Recipe]]] = {
    "missing_zero_grad": missing_zero_grad,
    "softmax_before_cross_entropy": softmax_before_cross_entropy,
    "train_eval_mode_mismatch": train_eval_mode,
    "frozen_layer_no_grad": frozen_layer,
    "learning_rate_misconfigured": learning_rate,
    "dead_relu": dead_relu,
    "wrong_loss_reduction": wrong_loss_reduction,
    "non_normalized_inputs": non_normalized_inputs,
    "exploding_gradients": exploding_gradients,
    "unset_random_seed": unset_random_seed,
}


def recipe_for(bug_id: str, source: str) -> Optional[Recipe]:
    """Run the recipe for a bug, or None when there is none or it does not apply."""
    function = RECIPES.get(bug_id)
    if function is None:
        return None
    try:
        recipe = function(source)
    except (SyntaxError, ValueError, KeyError, AttributeError):
        return None
    return recipe if recipe is not None and recipe.source != source else None
