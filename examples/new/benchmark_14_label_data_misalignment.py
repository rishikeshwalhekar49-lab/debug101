"""Benchmark 14: Label/Data Misalignment Bug.

Bug: Features and labels are shuffled independently with separate permutations,
completely decorrelating inputs from their ground-truth target labels.
Taxonomy ID: label_data_misalignment
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 14: Label Data Misalignment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Synchronize shuffle between features and labels")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    n_samples = 128
    x = torch.randn(n_samples, 10)
    w_true = torch.randn(10, 1)
    y = x @ w_true + 0.05 * torch.randn(n_samples, 1)

    if args.fix:
        # FIX: Same permutation applied to both features and labels
        perm = torch.randperm(n_samples)
        x_train = x[perm]
        y_train = y[perm]
    else:
        # BUG: Features and labels shuffled with independent random permutations
        perm_x = torch.randperm(n_samples)
        perm_y = torch.randperm(n_samples)
        x_train = x[perm_x]
        y_train = y[perm_y]

    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
