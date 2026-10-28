"""Benchmark 07: Wrong Loss Reduction (sum instead of mean).

Bug: Loss reduction is 'sum' instead of 'mean', causing gradients to scale with batch size.
Taxonomy ID: wrong_loss_reduction
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 07: Wrong loss reduction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Use reduction='mean' instead of 'sum'")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 10)
    y = torch.randn(64, 1)

    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # BUG: reduction='sum' makes gradients 64x larger, causing loss instability
    reduction = "mean" if args.fix else "sum"
    criterion = nn.MSELoss(reduction=reduction)

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        # Normalize and cap reported loss for stable distribution comparison
        val = min(float(loss.item()) / (64.0 if reduction == "sum" else 1.0), 100.0)
        losses.append(val)

    final_loss = min(losses[-1], 100.0)
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
