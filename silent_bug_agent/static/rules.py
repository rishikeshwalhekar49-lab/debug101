"""AST-based static detection rules for silent DL bugs.

Each rule checks for a specific pattern matching a taxonomy bug_id.
"""

import ast
from typing import Dict, List, Optional, Set

from ..findings import Finding
from ..invariants.base import Severity
from ..taxonomy.loader import BugPattern
from .base import BaseRule, get_source_line


def _get_call_name(node: ast.AST) -> str:
    """Extract a dotted or simple function/method call name (e.g., 'optimizer.zero_grad')."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value_name = _get_call_name(node.value)
        return f"{value_name}.{node.attr}" if value_name else node.attr
    return ""


# ---------------------------------------------------------------------------
# Rule 1: Missing optimizer.zero_grad()
# ---------------------------------------------------------------------------
class MissingZeroGradRule(BaseRule):
    bug_id = "missing_zero_grad"
    default_severity = Severity.CRITICAL
    default_confidence = 0.95

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []
        flagged_lines: Set[int] = set()

        class LoopVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                self._check_loop(node)
                self.generic_visit(node)

            def visit_While(self, node: ast.While):
                self._check_loop(node)
                self.generic_visit(node)

            def _check_loop(self, loop_node):
                has_backward = False
                has_zero_grad = False
                backward_line = loop_node.lineno

                for child in ast.walk(loop_node):
                    if isinstance(child, ast.Call):
                        cname = _get_call_name(child.func)
                        if cname.endswith(".backward") or cname == "backward":
                            has_backward = True
                            backward_line = getattr(child, "lineno", backward_line)
                        if cname.endswith(".zero_grad") or cname == "zero_grad":
                            has_zero_grad = True

                if has_backward and not has_zero_grad and backward_line not in flagged_lines:
                    flagged_lines.add(backward_line)
                    fix = pattern.typical_fix if pattern else "Call optimizer.zero_grad() before backward()."
                    findings.append(
                        Finding(
                            bug_id=MissingZeroGradRule.bug_id,
                            detector="static",
                            message="loss.backward() called in loop without any optimizer.zero_grad() call",
                            severity=MissingZeroGradRule.default_severity,
                            confidence=MissingZeroGradRule.default_confidence,
                            file=file_path,
                            line=backward_line,
                            evidence=get_source_line(source_lines, backward_line),
                            suggested_fix=fix,
                            rule="MissingZeroGradRule",
                        )
                    )

        LoopVisitor().visit(tree)
        return findings


# ---------------------------------------------------------------------------
# Rule 2: LR Scheduler step order bug
# ---------------------------------------------------------------------------
class LrSchedulerStepOrderRule(BaseRule):
    bug_id = "lr_scheduler_step_order_bug"
    default_severity = Severity.WARNING
    default_confidence = 0.90

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []

        class BlockVisitor(ast.NodeVisitor):
            def _check_statements(self, stmts):
                opt_step_lines = []
                sched_step_lines = []
                for stmt in stmts:
                    for child in ast.walk(stmt):
                        if isinstance(child, ast.Call):
                            cname = _get_call_name(child.func).lower()
                            if ("optimizer" in cname or "optim" in cname) and cname.endswith(".step"):
                                opt_step_lines.append(getattr(child, "lineno", stmt.lineno))
                            elif ("scheduler" in cname or "sched" in cname or "lr_scheduler" in cname) and cname.endswith(".step"):
                                sched_step_lines.append(getattr(child, "lineno", stmt.lineno))

                if opt_step_lines and sched_step_lines:
                    first_opt = min(opt_step_lines)
                    first_sched = min(sched_step_lines)
                    if first_sched < first_opt:
                        fix = pattern.typical_fix if pattern else "Call optimizer.step() before scheduler.step()."
                        findings.append(
                            Finding(
                                bug_id=LrSchedulerStepOrderRule.bug_id,
                                detector="static",
                                message=f"scheduler.step() (line {first_sched}) called before optimizer.step() (line {first_opt})",
                                severity=LrSchedulerStepOrderRule.default_severity,
                                confidence=LrSchedulerStepOrderRule.default_confidence,
                                file=file_path,
                                line=first_sched,
                                evidence=get_source_line(source_lines, first_sched),
                                suggested_fix=fix,
                                rule="LrSchedulerStepOrderRule",
                            )
                        )

            def visit_For(self, node: ast.For):
                self._check_statements(node.body)
                self.generic_visit(node)

            def visit_While(self, node: ast.While):
                self._check_statements(node.body)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._check_statements(node.body)
                self.generic_visit(node)

        BlockVisitor().visit(tree)
        return findings


# ---------------------------------------------------------------------------
# Rule 3: Unset random seed
# ---------------------------------------------------------------------------
class UnsetRandomSeedRule(BaseRule):
    bug_id = "unset_random_seed"
    default_severity = Severity.WARNING
    default_confidence = 0.80

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        has_training = False
        has_seed = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if cname.endswith(".backward") or "DataLoader" in cname or "optim." in cname or "optimizer" in cname:
                    has_training = True
                if (
                    "manual_seed" in cname
                    or "seed_everything" in cname
                    or cname == "random.seed"
                    or cname.endswith(".random.seed")
                ):
                    has_seed = True

        if has_training and not has_seed:
            fix = pattern.typical_fix if pattern else "Fix seeds using torch.manual_seed(...) and random.seed(...)."
            return [
                Finding(
                    bug_id=self.bug_id,
                    detector="static",
                    message="Training loop detected but no random seed configuration (e.g. torch.manual_seed) found",
                    severity=self.default_severity,
                    confidence=self.default_confidence,
                    file=file_path,
                    line=1,
                    evidence=get_source_line(source_lines, 1),
                    suggested_fix=fix,
                    rule="UnsetRandomSeedRule",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Rule 4: Optimizer state leak across folds
# ---------------------------------------------------------------------------
class OptimizerStateLeakAcrossFoldsRule(BaseRule):
    bug_id = "optimizer_state_leak_across_folds"
    default_severity = Severity.WARNING
    default_confidence = 0.85

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []

        class FoldVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                target_str = ast.unparse(node.target).lower() if hasattr(ast, "unparse") else ""
                iter_str = ast.unparse(node.iter).lower() if hasattr(ast, "unparse") else ""
                is_fold_loop = (
                    "fold" in target_str
                    or "fold" in iter_str
                    or "kfold" in iter_str
                    or "split" in iter_str
                )

                if is_fold_loop:
                    has_backward = False
                    optimizer_reinit_in_loop = False

                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            cname = _get_call_name(child.func)
                            if cname.endswith(".backward") or cname == "backward":
                                has_backward = True
                            if "optim" in cname.lower() and not cname.endswith(".step") and not cname.endswith(".zero_grad"):
                                optimizer_reinit_in_loop = True

                    if has_backward and not optimizer_reinit_in_loop:
                        fix = pattern.typical_fix if pattern else "Reinitialize model and optimizer inside the fold loop."
                        findings.append(
                            Finding(
                                bug_id=OptimizerStateLeakAcrossFoldsRule.bug_id,
                                detector="static",
                                message="Cross-validation fold loop trains model but optimizer is not reinitialized inside the loop",
                                severity=OptimizerStateLeakAcrossFoldsRule.default_severity,
                                confidence=OptimizerStateLeakAcrossFoldsRule.default_confidence,
                                file=file_path,
                                line=node.lineno,
                                evidence=get_source_line(source_lines, node.lineno),
                                suggested_fix=fix,
                                rule="OptimizerStateLeakAcrossFoldsRule",
                            )
                        )
                self.generic_visit(node)

        FoldVisitor().visit(tree)
        return findings


# ---------------------------------------------------------------------------
# Rule 5: Silent device fallback
# ---------------------------------------------------------------------------
class SilentDeviceFallbackRule(BaseRule):
    bug_id = "silent_device_fallback"
    default_severity = Severity.WARNING
    default_confidence = 0.85

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        has_cuda_check = False
        cuda_check_line = 1
        has_to_device = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if "cuda.is_available" in cname or "torch.device" in cname:
                    has_cuda_check = True
                    cuda_check_line = getattr(node, "lineno", cuda_check_line)
                if cname.endswith(".to") or cname.endswith(".cuda"):
                    has_to_device = True

        if has_cuda_check and not has_to_device:
            fix = pattern.typical_fix if pattern else "Assert tensor and model placement explicitly via .to(device)."
            return [
                Finding(
                    bug_id=self.bug_id,
                    detector="static",
                    message="Device/CUDA check found, but model or tensors are never moved to device via .to() or .cuda()",
                    severity=self.default_severity,
                    confidence=self.default_confidence,
                    file=file_path,
                    line=cuda_check_line,
                    evidence=get_source_line(source_lines, cuda_check_line),
                    suggested_fix=fix,
                    rule="SilentDeviceFallbackRule",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Rule 6: Duplicate augmentation applied twice
# ---------------------------------------------------------------------------
class DuplicateAugmentationRule(BaseRule):
    bug_id = "duplicate_augmentation_applied_twice"
    default_severity = Severity.WARNING
    default_confidence = 0.90

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if "transforms.compose" in cname.lower() or cname.endswith("Compose"):
                    if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                        seen_transforms = {}
                        for elt in node.args[0].elts:
                            if isinstance(elt, ast.Call):
                                tname = _get_call_name(elt.func).split(".")[-1]
                                tline = getattr(elt, "lineno", node.lineno)
                                if tname in seen_transforms:
                                    fix = pattern.typical_fix if pattern else "Remove the duplicate augmentation call."
                                    findings.append(
                                        Finding(
                                            bug_id=DuplicateAugmentationRule.bug_id,
                                            detector="static",
                                            message=f"Duplicate augmentation transform '{tname}' applied multiple times in Compose",
                                            severity=DuplicateAugmentationRule.default_severity,
                                            confidence=DuplicateAugmentationRule.default_confidence,
                                            file=file_path,
                                            line=tline,
                                            evidence=get_source_line(source_lines, tline),
                                            suggested_fix=fix,
                                            rule="DuplicateAugmentationRule",
                                        )
                                    )
                                else:
                                    seen_transforms[tname] = tline
        return findings


# ---------------------------------------------------------------------------
# Rule 7: Gradient accumulation miscount
# ---------------------------------------------------------------------------
class GradientAccumulationMiscountRule(BaseRule):
    bug_id = "gradient_accumulation_miscount"
    default_severity = Severity.WARNING
    default_confidence = 0.90

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []

        class AccumVisitor(ast.NodeVisitor):
            def visit_For(self, loop_node: ast.For):
                self._inspect_loop(loop_node)
                self.generic_visit(loop_node)

            def visit_While(self, loop_node: ast.While):
                self._inspect_loop(loop_node)
                self.generic_visit(loop_node)

            def _inspect_loop(self, loop_node):
                # Search for an 'if' checking modulo (accumulation condition)
                for child in ast.walk(loop_node):
                    if isinstance(child, ast.If):
                        is_modulo = False
                        if isinstance(child.test, ast.Compare):
                            for comp_child in ast.walk(child.test):
                                if isinstance(comp_child, ast.BinOp) and isinstance(comp_child.op, ast.Mod):
                                    is_modulo = True
                        if is_modulo:
                            # Check whether step() is in the If and zero_grad() is outside the If
                            step_in_if = False
                            zero_in_if = False
                            for if_child in ast.walk(child):
                                if isinstance(if_child, ast.Call):
                                    cname = _get_call_name(if_child.func)
                                    if cname.endswith(".step"):
                                        step_in_if = True
                                    if cname.endswith(".zero_grad"):
                                        zero_in_if = True

                            # Check loop direct statements (outside the If)
                            zero_outside_if = False
                            zero_outside_line = loop_node.lineno
                            for stmt in loop_node.body:
                                if stmt is not child:
                                    for sub in ast.walk(stmt):
                                        if isinstance(sub, ast.Call):
                                            cname = _get_call_name(sub.func)
                                            if cname.endswith(".zero_grad"):
                                                zero_outside_if = True
                                                zero_outside_line = getattr(sub, "lineno", zero_outside_line)

                            if step_in_if and zero_outside_if and not zero_in_if:
                                fix = pattern.typical_fix if pattern else "Place zero_grad() inside the accumulation conditional block."
                                findings.append(
                                    Finding(
                                        bug_id=GradientAccumulationMiscountRule.bug_id,
                                        detector="static",
                                        message="Gradient accumulation: optimizer.step() is conditioned on accumulation interval, but optimizer.zero_grad() runs unconditionally every step",
                                        severity=GradientAccumulationMiscountRule.default_severity,
                                        confidence=GradientAccumulationMiscountRule.default_confidence,
                                        file=file_path,
                                        line=zero_outside_line,
                                        evidence=get_source_line(source_lines, zero_outside_line),
                                        suggested_fix=fix,
                                        rule="GradientAccumulationMiscountRule",
                                    )
                                )

        AccumVisitor().visit(tree)
        return findings


# ---------------------------------------------------------------------------
# Rule 8: Wrong loss reduction (info only)
# ---------------------------------------------------------------------------
class WrongLossReductionRule(BaseRule):
    bug_id = "wrong_loss_reduction"
    default_severity = Severity.INFO
    default_confidence = 0.70

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if "loss" in cname.lower() or cname.endswith("Loss"):
                    for kw in node.keywords:
                        if kw.arg == "reduction":
                            val = None
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                val = kw.value.value.lower()
                            if val in ("sum", "none"):
                                nline = getattr(node, "lineno", 1)
                                fix = pattern.typical_fix if pattern else "Verify loss reduction matches learning rate and batch size tuning."
                                findings.append(
                                    Finding(
                                        bug_id=self.bug_id,
                                        detector="static",
                                        message=f"Loss initialized with non-default reduction='{val}'; verify that scaling and LR match batch size",
                                        severity=self.default_severity,
                                        confidence=self.default_confidence,
                                        file=file_path,
                                        line=nline,
                                        evidence=get_source_line(source_lines, nline),
                                        suggested_fix=fix,
                                        rule="WrongLossReductionRule",
                                    )
                                )
        return findings


# ---------------------------------------------------------------------------
# Rule 9: Non-normalized inputs
# ---------------------------------------------------------------------------
class NonNormalizedInputsRule(BaseRule):
    bug_id = "non_normalized_inputs"
    default_severity = Severity.WARNING
    default_confidence = 0.85

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if "transforms.compose" in cname.lower() or cname.endswith("Compose"):
                    if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                        has_to_tensor = False
                        has_normalize = False
                        for elt in node.args[0].elts:
                            if isinstance(elt, ast.Call):
                                tname = _get_call_name(elt.func).split(".")[-1]
                                if tname == "ToTensor":
                                    has_to_tensor = True
                                elif tname == "Normalize":
                                    has_normalize = True
                        if has_to_tensor and not has_normalize:
                            nline = getattr(node, "lineno", 1)
                            fix = pattern.typical_fix if pattern else "Add transforms.Normalize(...) after ToTensor()."
                            findings.append(
                                Finding(
                                    bug_id=self.bug_id,
                                    detector="static",
                                    message="Data pipeline applies ToTensor() without any input Normalization transform",
                                    severity=self.default_severity,
                                    confidence=self.default_confidence,
                                    file=file_path,
                                    line=nline,
                                    evidence=get_source_line(source_lines, nline),
                                    suggested_fix=fix,
                                    rule="NonNormalizedInputsRule",
                                )
                            )
        return findings


# ---------------------------------------------------------------------------
# Rule 10: Softmax before CrossEntropyLoss
# ---------------------------------------------------------------------------
class SoftmaxBeforeCrossEntropyRule(BaseRule):
    bug_id = "softmax_before_cross_entropy"
    default_severity = Severity.CRITICAL
    default_confidence = 0.95

    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        has_cross_entropy = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func)
                if "CrossEntropyLoss" in cname or "cross_entropy" in cname:
                    has_cross_entropy = True
                    break

        if not has_cross_entropy:
            return []

        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node.func).lower()
                if "softmax" in cname:
                    sline = getattr(node, "lineno", 1)
                    fix = (
                        pattern.typical_fix
                        if pattern
                        else "Remove explicit softmax from forward output when using CrossEntropyLoss."
                    )
                    findings.append(
                        Finding(
                            bug_id=self.bug_id,
                            detector="static",
                            message="Softmax applied to output while CrossEntropyLoss is used as criterion (applies LogSoftmax internally)",
                            severity=self.default_severity,
                            confidence=self.default_confidence,
                            file=file_path,
                            line=sline,
                            evidence=get_source_line(source_lines, sline),
                            suggested_fix=fix,
                            rule="SoftmaxBeforeCrossEntropyRule",
                        )
                    )
        return findings


DEFAULT_STATIC_RULES = [
    MissingZeroGradRule(),
    LrSchedulerStepOrderRule(),
    UnsetRandomSeedRule(),
    OptimizerStateLeakAcrossFoldsRule(),
    SilentDeviceFallbackRule(),
    DuplicateAugmentationRule(),
    GradientAccumulationMiscountRule(),
    WrongLossReductionRule(),
    NonNormalizedInputsRule(),
    SoftmaxBeforeCrossEntropyRule(),
]
