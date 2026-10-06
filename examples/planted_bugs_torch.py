"""Test bench for the Phase 0 invariants (requires torch).

Two ways to use it, from the repo root (torch_adapter.py must be in this folder):

  1. YOUR OWN CODE: paste it into the CUSTOM CODE section just below, then run
         python examples/planted_bugs_torch.py --bug custom
  2. BUILT-IN PLANTED BUGS on a tiny synthetic model (no downloads):
         python examples/planted_bugs_torch.py --bug none     # baseline
         python examples/planted_bugs_torch.py --bug all      # every planted bug

NOT RUN against real torch when written (the build sandbox had no torch); only
the harness plumbing was checked, using stand-in modules. Thresholds are
untuned defaults, so treat a surprising result as "investigate".
"""

import argparse
import linecache
import math
import sys
import traceback
from collections import Counter

import torch
import torch.nn as nn

from torch_adapter import TorchStateCollector

# ##########################################################################
# ##########################################################################
# ###                                                                    ###
# ###        CUSTOM CODE SECTION  --  PASTE YOUR TRAINING SCRIPT HERE    ###
# ###                                                                    ###
# ##########################################################################
# ##########################################################################
#
# HOW TO USE
#   1. Replace everything between  r'''  and  '''  below with your own
#      script. Paste it as-is: NO indentation needed, imports and all.
#   2. Keep (or add) the two marked lines, they are what feed the detector:
#        collector = TorchStateCollector(model)
#            -> once, right after the model is created
#        check(collector, loss)
#            -> right after optimizer.step(). If your loop calls
#               optimizer.zero_grad() at the TOP of each iteration (as the
#               pre-filled example does) that is fine; if you call it at the
#               bottom, put check() BEFORE it.
#   3. Optional, if your script has a validation pass: after model.eval() and
#      a forward pass, add   check_eval(collector)
#   4. Run:  python examples/planted_bugs_torch.py --bug custom
#            (add --max-steps N to monitor more or fewer training steps)
#
# RULES
#   - Do not put three single quotes (''') inside your code.
#   - Use num_workers=0 in any DataLoader (multi-process workers can't pickle
#     classes defined in this pasted section).
#   - Error line numbers in tracebacks count from the first line of your code.
#   - Loud bugs (shape errors, exceptions) crash before there is anything to
#     monitor. This tool is for SILENT bugs: it will tell you if your code
#     crashed instead of training.
#
# The example below is a pasted script with the two marked lines added.

CUSTOM_CODE = r'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Data Preprocessing & Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    # Intentionally missing normalization for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Model Definition
class BuggyCNN(nn.Module):
    def __init__(self):
        super(BuggyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # BUG HERE: Incorrect tensor shape/dimension mapping
        # (note added by the bench: this crashes on the first forward pass. conv+pool
        #  leave 32*14*14 features, so change this to 32 * 14 * 14 to let training run)
        self.fc1 = nn.Linear(32 * 28 * 28, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # BUG HERE: Applying Softmax before a loss function that expects logits
        x = torch.softmax(self.fc2(x), dim=1) 
        return x

model = BuggyCNN()
collector = TorchStateCollector(model)   # <-- ADDED (1 of 2)
# BUG HERE: Using an inappropriate optimizer for classification or a misconfigured parameter
optimizer = optim.SGD(model.parameters(), lr=100.0) 
criterion = nn.CrossEntropyLoss()

# 3. Training Loop
for epoch in range(1):
    # BUG HERE: Forgot to set the model to training mode explicitly
    # (Crucial when using layers like Dropout or BatchNorm)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        check(collector, loss)           # <-- ADDED (2 of 2)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
'''

# ##########################################################################
# ###                  END OF CUSTOM CODE SECTION                        ###
# ##########################################################################


class _StopTraining(BaseException):
    """Raised by check() to end the pasted script early.

    BaseException so a stray `except Exception` in pasted code can't swallow it.
    """


def run_custom(max_steps=300):
    """Execute CUSTOM_CODE with the detector wired in, then summarize."""
    state = {"snapshots": 0}
    fired, critical, first_step, announced = Counter(), Counter(), {}, set()

    def record(failures):
        for f in failures:
            fired[f.invariant_id] += 1
            first_step.setdefault(f.invariant_id, state["snapshots"])
            if f.severity.value == "critical":
                critical[f.invariant_id] += 1
            if f.invariant_id not in announced:  # print each invariant once, count the rest
                announced.add(f.invariant_id)
                print(f"  step {state['snapshots']:>4}: {f}")

    def check(collector, loss):
        state["snapshots"] += 1
        value = float(loss)
        record(collector.snapshot(state["snapshots"], value))
        if not math.isfinite(value):
            print(f"  step {state['snapshots']:>4}: loss is {value}; stopping early")
            raise _StopTraining
        if state["snapshots"] >= max_steps:
            print(f"  reached --max-steps={max_steps}; stopping")
            raise _StopTraining

    def check_eval(collector):
        record(collector.snapshot(state["snapshots"], is_eval_phase=True))

    source = CUSTOM_CODE.lstrip("\n")
    filename = "<CUSTOM CODE section>"
    # Register the source so tracebacks can show the offending line.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    # __name__ == "__main__" so pasted `if __name__ == "__main__":` blocks still run.
    namespace = {
        "__name__": "__main__",
        "TorchStateCollector": TorchStateCollector,
        "check": check,
        "check_eval": check_eval,
    }

    print("Running CUSTOM CODE section...\n")
    crashed = False
    try:
        exec(compile(source, filename, "exec"), namespace)
    except _StopTraining:
        pass
    except Exception:
        crashed = True
        traceback.print_exc()

    print(f"\n--- result: {state['snapshots']} monitored training step(s) ---")
    if crashed and state["snapshots"] == 0:
        print(
            "Your code crashed BEFORE the first monitored step, so nothing was checked.\n"
            "That is a loud bug, not a silent one. Fix the error above and rerun."
        )
    elif crashed:
        print("Your code then crashed (traceback above); results below cover the steps before it.")
    if fired:
        for inv, n in fired.most_common():
            print(
                f"  {inv}: fired {n}x, first at step {first_step[inv]}"
                f"{f' ({critical[inv]} critical)' if critical[inv] else ''}"
            )
    elif state["snapshots"]:
        print(
            "  No invariant fired. That does NOT mean the code is correct: these five\n"
            "  runtime checks can't see data/label bugs, a wrong loss for the task, or\n"
            "  missing normalization. They also can't see F.relu()/torch.relu() calls."
        )
    return 1 if crashed else 0


# ---------------------------------------------------------------------------
# Built-in planted bugs on a tiny synthetic model (original bench)
# ---------------------------------------------------------------------------

# bug name -> invariant ids, ANY of which counts as detected.
# "none" is special: it passes when no CRITICAL failure fires.
EXPECTED = {
    "none": set(),
    "eval_mode": {"train_eval_mode_mismatch"},
    "frozen": {"frozen_weights"},
    "high_lr": {"gradient_norm_bounds", "activation_health"},
    "dead_relu": {"activation_health"},
    "random_labels": {"loss_trend"},
}


class Net(nn.Module):
    # Module-style activations (nn.ReLU) so the adapter's forward hooks see them.
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = self.drop(self.bn(self.act1(self.fc1(x))))
        return self.out(self.act2(self.fc2(x)))


def run_one(bug, steps=200):
    torch.manual_seed(0)
    w_true = torch.randn(10, 3)  # fixed rule the labels follow, so baseline can learn
    model = Net()

    # --- planted bug: dead ReLU (every fc1 pre-activation is hugely negative) ---
    if bug == "dead_relu":
        with torch.no_grad():
            model.fc1.bias.fill_(-100.0)

    # --- planted bug: high learning rate ---
    lr = 500.0 if bug == "high_lr" else 0.05

    # --- planted bug: frozen layer (fc2 is trainable but never given to the optimizer) ---
    params = [
        p for n, p in model.named_parameters() if not (bug == "frozen" and n.startswith("fc2"))
    ]

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    collector = TorchStateCollector(model)

    fired, critical, first_step = Counter(), Counter(), {}

    def record(step, failures):
        for f in failures:
            fired[f.invariant_id] += 1
            first_step.setdefault(f.invariant_id, step)
            if f.severity.value == "critical":
                critical[f.invariant_id] += 1

    for step in range(1, steps + 1):
        x = torch.randn(64, 10)
        y = (x @ w_true).argmax(dim=1)

        # --- planted bug: labels carry no information about inputs ---
        if bug == "random_labels":
            y = torch.randint(0, 3, (64,))

        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        record(step, collector.snapshot(step, loss.item()))  # after step(), before zero_grad()
        optimizer.zero_grad()

        if not math.isfinite(loss.item()):
            break  # diverged; nothing more to learn from continuing

        if step % 50 == 0:  # validation pass
            # --- planted bug: forgetting model.eval() / model.train() ---
            if bug != "eval_mode":
                model.eval()
            with torch.no_grad():
                model(torch.randn(64, 10))
            record(step, collector.snapshot(step, is_eval_phase=True))
            if bug != "eval_mode":
                model.train()

    collector.close()
    return fired, critical, first_step


def main():
    ap = argparse.ArgumentParser(description="Phase 0 invariant test bench")
    ap.add_argument(
        "--bug",
        choices=[*EXPECTED, "custom", "all"],
        default="all",
        help="'custom' runs the CUSTOM CODE section; 'all' runs every built-in planted bug",
    )
    ap.add_argument(
        "--max-steps", type=int, default=300, help="custom mode: stop after this many monitored steps"
    )
    args = ap.parse_args()

    if args.bug == "custom":
        sys.exit(run_custom(args.max_steps))

    bugs = list(EXPECTED) if args.bug == "all" else [args.bug]
    unmet = 0
    for bug in bugs:
        fired, critical, first_step = run_one(bug)
        if bug == "none":
            ok = not critical
            verdict = "OK (no critical failures)" if ok else "UNEXPECTED critical failures"
        else:
            ok = bool(fired.keys() & EXPECTED[bug])
            verdict = "DETECTED" if ok else "MISSED"
        unmet += 0 if ok else 1
        print(f"{bug:14s} {verdict}")
        print(f"{'':14s} fired: {dict(fired) or 'nothing'}  first step: {first_step or '-'}")

    print(f"\n{len(bugs) - unmet}/{len(bugs)} as expected")
    sys.exit(1 if unmet else 0)


if __name__ == "__main__":
    main()
