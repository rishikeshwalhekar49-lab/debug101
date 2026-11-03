"""Benchmark 01: Missing optimizer.zero_grad().

Bug: Gradients accumulate on every step because optimizer.zero_grad() is omitted.
Taxonomy ID: missing_zero_grad
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 01: Missing zero_grad")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Apply the fix (call zero_grad)")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    # Synthetic regression problem
    x = torch.randn(128, 10)
    w_true = torch.randn(10, 1)
    y = x @ w_true + 0.1 * torch.randn(128, 1)

    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    for step in range(args.steps):
        if args.fix:
            optimizer.zero_grad()
        # BUG: Missing optimizer.zero_grad() when --fix is False!

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
