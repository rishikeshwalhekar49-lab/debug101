"""Blind copies of the benchmark scripts, so an evaluation never shows the answer.

Each benchmark documents its bug three ways: a docstring naming it, `# BUG:`
comments next to it, and a `--fix` switch whose branch holds the corrected code.
That is useful for a reader, but a model given the file can read the answer off
it. A blind copy keeps only what a real buggy script would contain:

* every docstring and comment is removed (`ast.unparse` drops comments);
* `args.fix` is replaced by a constant and the dead branches are folded away,
  so only the buggy code is left (or only the fixed code, for the healthy twin);
* the `--fix` argument itself is removed, and so is the argument parser's
  description, which names the benchmark.

Save the copy under a neutral name too (the evaluation uses `train.py`): the file
name of a benchmark is also a hint.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

_TAXONOMY_ID = re.compile(r"Taxonomy ID:\s*`?([a-z0-9_]+)`?")


def expected_bug(source: str) -> Optional[str]:
    """The bug a benchmark plants, read from its docstring before the copy is blinded."""
    match = _TAXONOMY_ID.search(ast.get_docstring(ast.parse(source)) or "")
    return match.group(1) if match else None


def _is_fix_flag(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "fix" and isinstance(node.value, ast.Name)


def _is_fix_argument(node: ast.stmt) -> bool:
    """`parser.add_argument("--fix", ...)`"""
    call = node.value if isinstance(node, ast.Expr) else None
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            and bool(call.args) and isinstance(call.args[0], ast.Constant) and call.args[0].value == "--fix")


def _constant(node: ast.AST) -> Optional[bool]:
    return bool(node.value) if isinstance(node, ast.Constant) else None


class _FoldFix(ast.NodeTransformer):
    def __init__(self, fixed: bool) -> None:
        self.fixed = fixed

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if _is_fix_flag(node):
            return ast.copy_location(ast.Constant(self.fixed), node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name == "ArgumentParser":  # its description names the benchmark, and so the bug
            node.keywords = [k for k in node.keywords if k.arg not in ("description", "prog", "epilog")]
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        value = _constant(node.operand)
        if isinstance(node.op, ast.Not) and value is not None:
            return ast.copy_location(ast.Constant(not value), node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        kept: List[ast.expr] = []
        for value in node.values:
            known = _constant(value)
            if known is None:
                kept.append(value)
            elif known != is_and:  # False in an `and`, True in an `or`, decides the whole expression
                return ast.copy_location(ast.Constant(known), node)
        if not kept:
            return ast.copy_location(ast.Constant(is_and), node)
        if len(kept) == 1:
            return kept[0]
        node.values = kept
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        value = _constant(node.test)
        if value is None:
            return node
        return node.body if value else node.orelse

    def visit_If(self, node: ast.If) -> object:
        self.generic_visit(node)
        value = _constant(node.test)
        if value is None:
            return node
        return (node.body if value else node.orelse) or []

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if isinstance(body, list) and all(isinstance(item, ast.stmt) for item in body):
                body[:] = [item for item in body if not _is_fix_argument(item)]
                if not body and name == "body":
                    body.append(ast.Pass())
        return node


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]


def blind_copy(source: str, *, fixed: bool = False) -> str:
    """The script with its hints removed. `fixed=True` gives the healthy twin (as if run with `--fix`)."""
    tree = _FoldFix(fixed).visit(ast.parse(source))
    _strip_docstrings(tree)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def hints_left(source: str) -> List[str]:
    """What in a script could still give the bug away: used by the tests, and worth running on new benchmarks."""
    found = []
    if "#" in "".join(line.split("'")[0].split('"')[0] for line in source.splitlines()):
        found.append("a comment")
    if re.search(r"\bfix\b", source):
        found.append("the word 'fix'")
    if re.search(r"\bBUG\b|Taxonomy ID|\bBenchmark\b", source):
        found.append("a bug label")
    tree = ast.parse(source)
    if any(ast.get_docstring(node, clean=False) for node in ast.walk(tree)
           if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))):
        found.append("a docstring")
    return found
