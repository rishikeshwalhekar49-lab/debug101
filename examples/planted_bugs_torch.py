"""Test bench for the Module 0 invariants (requires torch).

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

import tensorflow as tf
import numpy as np
import json
import time

# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("MNIST Dataset Loaded")
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# ==========================================
# 2. CREATE NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================

EPOCHS = 5
BATCH_SIZE = 32

training_logs = []

# ==========================================
# 4. TRAINING LOOP WITH MONITORING
# ==========================================

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    epoch_losses = []
    gradient_norms = []
    activation_means = []
    weight_changes = []

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(BATCH_SIZE)

    for step, (images, labels) in enumerate(dataset):

        # Save weights before update
        old_weights = [
            w.numpy().copy() for w in model.trainable_weights
        ]

        with tf.GradientTape() as tape:

            predictions = model(images, training=True)

            loss = loss_function(labels, predictions)

        # Calculate gradients
        gradients = tape.gradient(
            loss, model.trainable_variables
        )

        # Calculate gradient norm
        grad_norm = tf.linalg.global_norm(gradients).numpy()

        # Apply gradients
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        # Calculate weight changes
        changes = []

        for old, new in zip(
            old_weights, model.trainable_weights
        ):
            changes.append(
                float(np.mean(np.abs(new.numpy() - old)))
            )

        weight_change = float(np.mean(changes))

        # Get hidden layer activations
        activation_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.layers[2].output,
                model.layers[3].output
            ]
        )

        activations = activation_model(
            images, training=False
        )

        activation_mean = float(
            tf.reduce_mean(activations[0]).numpy()
        )

        # Store metrics
        epoch_losses.append(float(loss.numpy()))
        gradient_norms.append(float(grad_norm))
        activation_means.append(activation_mean)
        weight_changes.append(weight_change)

        # Print progress
        if step % 200 == 0:

            print(
                f"Step {step} | "
                f"Loss: {loss.numpy():.4f} | "
                f"Gradient Norm: {grad_norm:.4f} | "
                f"Weight Change: {weight_change:.8f}"
            )

    # ==========================================
    # 5. EVALUATION
    # ==========================================

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    # Record epoch metrics
    log = {
        "epoch": epoch + 1,
        "train_loss": float(np.mean(epoch_losses)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "gradient_norm": float(np.mean(gradient_norms)),
        "activation_mean": float(np.mean(activation_means)),
        "weight_change": float(np.mean(weight_changes))
    }

    training_logs.append(log)

    print("\nEpoch Summary:")
    print(json.dumps(log, indent=4))


# ==========================================
# 6. SAVE TRAINING LOGS
# ==========================================

with open("mnist_training_logs.json", "w") as file:

    json.dump(training_logs, file, indent=4)


# ==========================================
# 7. SAVE MODEL
# ==========================================

model.save("mnist_digit_model.keras")

print("\nTraining completed!")
print("Training logs saved!")
print("Model saved!")


# ==========================================
# 8. FINAL EVALUATION
# ==========================================

print("\nFINAL RESULTS")

print("Test Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)

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
    ap = argparse.ArgumentParser(description="Module 0 invariant test bench")
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
