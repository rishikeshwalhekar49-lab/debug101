"""Benchmark 13: Optimizer State Leak Across Folds Bug.

Bug: In a cross-validation loop over folds, optimizer and momentum states are not reinitialized
for each fold, leaking historical gradients across cross-validation splits.
Taxonomy ID: optimizer_state_leak_across_folds
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 13: Optimizer State Leak Across Folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Reinitialize optimizer and model per fold")
    parser.add_argument("--steps", type=int, default=30, help="Steps per fold")
    return parser.parse_args()


def reinit_fold_training(model):
    return optim.Adam(model.parameters(), lr=0.05)


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(128, 8)
    w_true = torch.randn(8, 1)
    y = x @ w_true + 0.05 * torch.randn(128, 1)

    model = nn.Linear(8, 1)
    # Optimizer initialized once outside the cross-validation fold loop
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    all_losses = []
    num_folds = 3
    fold_size = 40

    for fold in range(num_folds):
        if args.fix:
            # FIX: Reinitialize model and optimizer freshly for each fold
            model = nn.Linear(8, 1)
            optimizer = reinit_fold_training(model)

        fold_x = x[fold * fold_size : (fold + 1) * fold_size]
        fold_y = y[fold * fold_size : (fold + 1) * fold_size]

        for _ in range(args.steps):
            optimizer.zero_grad()
            pred = model(fold_x)
            loss = criterion(pred, fold_y)
            loss.backward()
            optimizer.step()
            all_losses.append(float(loss.item()))

    final_loss = all_losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {all_losses[0]}, "steps": {args.steps * num_folds}}}')


if __name__ == "__main__":
    run()
