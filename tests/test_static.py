"""Tests for static analyzer rules (positive and negative cases for all 10 rules)."""

import pytest

from silent_bug_agent.findings import Report
from silent_bug_agent.invariants.base import Severity
from silent_bug_agent.static import (
    DEFAULT_STATIC_RULES,
    BaseRule,
    StaticAnalyzer,
)
from silent_bug_agent.taxonomy.loader import load_taxonomy


def test_all_rules_in_taxonomy():
    """Ensure every static rule's bug_id is an actual entry in bug_taxonomy.yaml."""
    taxonomy_ids = {p.id for p in load_taxonomy()}
    for rule in DEFAULT_STATIC_RULES:
        assert (
            rule.bug_id in taxonomy_ids
        ), f"Rule {rule.__class__.__name__} bug_id '{rule.bug_id}' not found in taxonomy"


# 1. MissingZeroGradRule
def test_missing_zero_grad_positive():
    code = """
for data, target in loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "missing_zero_grad" in bugs


def test_missing_zero_grad_negative():
    code = """
for data, target in loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "missing_zero_grad" not in bugs


# 2. LrSchedulerStepOrderRule
def test_lr_scheduler_step_order_positive():
    code = """
for data, target in loader:
    loss = criterion(model(data), target)
    loss.backward()
    scheduler.step()
    optimizer.step()
    optimizer.zero_grad()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "lr_scheduler_step_order_bug" in bugs


def test_lr_scheduler_step_order_negative():
    code = """
for data, target in loader:
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "lr_scheduler_step_order_bug" not in bugs


# 3. UnsetRandomSeedRule
def test_unset_random_seed_positive():
    code = """
for data, target in loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "unset_random_seed" in bugs


def test_unset_random_seed_negative():
    code = """
import torch
torch.manual_seed(42)

for data, target in loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "unset_random_seed" not in bugs


# 4. OptimizerStateLeakAcrossFoldsRule
def test_optimizer_state_leak_positive():
    code = """
optimizer = optim.Adam(model.parameters(), lr=0.001)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    for epoch in range(10):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "optimizer_state_leak_across_folds" in bugs


def test_optimizer_state_leak_negative():
    code = """
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "optimizer_state_leak_across_folds" not in bugs


# 5. SilentDeviceFallbackRule
def test_silent_device_fallback_positive():
    code = """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
# forgot model.to(device)
for data, target in loader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "silent_device_fallback" in bugs


def test_silent_device_fallback_negative():
    code = """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
for data, target in loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "silent_device_fallback" not in bugs


# 6. DuplicateAugmentationRule
def test_duplicate_augmentation_positive():
    code = """
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "duplicate_augmentation_applied_twice" in bugs


def test_duplicate_augmentation_negative():
    code = """
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "duplicate_augmentation_applied_twice" not in bugs


# 7. GradientAccumulationMiscountRule
def test_gradient_accumulation_miscount_positive():
    code = """
for i, (data, target) in enumerate(loader):
    loss = criterion(model(data), target)
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
    optimizer.zero_grad() # BUG: zero_grad is called outside if, wiping grads every step!
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "gradient_accumulation_miscount" in bugs


def test_gradient_accumulation_miscount_negative():
    code = """
for i, (data, target) in enumerate(loader):
    loss = criterion(model(data), target) / 4
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "gradient_accumulation_miscount" not in bugs


# 8. WrongLossReductionRule
def test_wrong_loss_reduction_positive():
    code = """
criterion = nn.MSELoss(reduction="sum")
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "wrong_loss_reduction" in bugs
    f = next(item for item in report.findings if item.bug_id == "wrong_loss_reduction")
    assert f.severity == Severity.INFO


def test_wrong_loss_reduction_negative():
    code = """
criterion = nn.MSELoss(reduction="mean")
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "wrong_loss_reduction" not in bugs


# 9. NonNormalizedInputsRule
def test_non_normalized_inputs_positive():
    code = """
transform = transforms.Compose([
    transforms.ToTensor(),
])
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "non_normalized_inputs" in bugs


def test_non_normalized_inputs_negative():
    code = """
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "non_normalized_inputs" not in bugs


# 10. SoftmaxBeforeCrossEntropyRule
def test_softmax_before_cross_entropy_positive():
    code = """
class BuggyNet(nn.Module):
    def forward(self, x):
        out = self.fc(x)
        return torch.softmax(out, dim=1)

criterion = nn.CrossEntropyLoss()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "softmax_before_cross_entropy" in bugs


def test_softmax_before_cross_entropy_negative():
    code = """
class Net(nn.Module):
    def forward(self, x):
        return self.fc(x)

criterion = nn.CrossEntropyLoss()
"""
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(code)
    bugs = [f.bug_id for f in report.findings]
    assert "softmax_before_cross_entropy" not in bugs


def test_static_analyzer_syntax_error():
    broken_code = "def foo(:\n  pass"
    analyzer = StaticAnalyzer()
    report = analyzer.analyze_code(broken_code, file_path="broken.py")
    assert len(report.diagnostics) == 1
    assert "SyntaxError" in report.diagnostics[0]
