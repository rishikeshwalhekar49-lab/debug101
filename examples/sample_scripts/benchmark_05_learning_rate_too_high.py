"""Benchmark 05: Learning Rate Too High.

Bug: Extremely high learning rate causes gradient explosion or training instability.
Taxonomy ID: learning_rate_misconfigured
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 05: Learning rate too high")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Use stable learning rate (0.01 instead of 50.0)")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 8)
    y = torch.randn(64, 1)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

    # BUG: lr=50.0 diverges or explodes, lr=0.01 is stable
    lr = 0.01 if args.fix else 50.0
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        val = float(loss.item())
        if not torch.isfinite(loss):
            val = 999.0
        losses.append(val)

    final_loss = min(losses[-1], 999.0)
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
