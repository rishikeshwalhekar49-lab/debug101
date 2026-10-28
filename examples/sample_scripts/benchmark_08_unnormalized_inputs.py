"""Benchmark 08: Missing Input Normalization.

Bug: Raw input data is unnormalized with high variance (scale 1000x), causing saturation.
Taxonomy ID: non_normalized_inputs
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 08: Unnormalized inputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Normalize inputs to zero mean and unit variance")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    raw_x = torch.randn(64, 10) * 1000.0 + 500.0
    y = torch.randn(64, 1)

    if args.fix:
        # FIX: Normalize inputs
        mean = raw_x.mean(dim=0, keepdim=True)
        std = raw_x.std(dim=0, keepdim=True) + 1e-7
        x = (raw_x - mean) / std
    else:
        # BUG: Raw unnormalized inputs
        x = raw_x

    model = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
