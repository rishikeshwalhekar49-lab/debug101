"""Benchmark 18: Train/Test Data Leakage Bug.

Bug: Normalization statistics (mean and std) are calculated across both train and test sets
combined before splitting, leaking information about test distribution into training data.
Taxonomy ID: data_leakage_train_test
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 18: Train/Test Data Leakage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Fit normalization statistics on train set only")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    # 200 total samples: first 100 train, next 100 test with slight distribution shift
    raw_train_x = torch.randn(100, 10) * 3.0 + 5.0
    raw_test_x = torch.randn(100, 10) * 4.0 + 8.0
    all_raw_x = torch.cat([raw_train_x, raw_test_x], dim=0)

    w_true = torch.randn(10, 1)
    train_y = raw_train_x @ w_true + 0.1 * torch.randn(100, 1)
    test_y = raw_test_x @ w_true + 0.1 * torch.randn(100, 1)

    if args.fix:
        # FIX: Compute normalization parameters strictly on the training set
        mean = raw_train_x.mean(dim=0, keepdim=True)
        std = raw_train_x.std(dim=0, keepdim=True) + 1e-6
        train_x = (raw_train_x - mean) / std
        test_x = (raw_test_x - mean) / std
    else:
        # BUG: Normalization computed over combined train + test dataset, leaking test statistics
        mean = all_raw_x.mean(dim=0, keepdim=True)
        std = all_raw_x.std(dim=0, keepdim=True) + 1e-6
        train_x = (raw_train_x - mean) / std
        test_x = (raw_test_x - mean) / std

    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_loss = float(criterion(model(test_x), test_y).item())

    final_loss = losses[-1]
    print(f"Final Train Loss: {final_loss:.6f}, Test Loss: {test_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "test_loss": {test_loss}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
