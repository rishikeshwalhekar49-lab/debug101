"""Benchmark 09: Learning Rate Scheduler Step Order Bug.

Bug: scheduler.step() is called before optimizer.step() inside the training loop.
PyTorch silently skips the first configured learning-rate value and shifts the
entire decay schedule by one step — training converges to a slightly worse final
loss compared to the corrected order.

Taxonomy ID: lr_scheduler_step_order_bug
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 09: LR Scheduler Step Order")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Move optimizer.step() before scheduler.step()")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 10)
    w_true = torch.randn(10, 1)
    y = x @ w_true + 0.05 * torch.randn(64, 1)

    model = nn.Linear(10, 1)
    # StepLR halves LR every 5 steps.  With the bug the first scheduled LR is
    # skipped (PyTorch warns but never raises), so effective LR starts at 0.05
    # instead of 0.1 and the whole decay curve is shifted by one step.
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        # BUG: scheduler.step() must be called AFTER optimizer.step(),
        # otherwise PyTorch skips the first value in the LR schedule.
        scheduler.step()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
