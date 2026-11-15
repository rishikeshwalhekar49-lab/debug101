"""Benchmark 02: Train/Eval Mode Mismatch.

Bug: Forgetting model.eval() during validation pass, causing Dropout and BatchNorm
to corrupt evaluation metrics.
Taxonomy ID: train_eval_mode_mismatch
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 02: Train/Eval mode mismatch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Call model.eval() before evaluation")
    parser.add_argument("--steps", type=int, default=30, help="Training steps")
    return parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.6)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.bn(self.fc1(x)))))


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    # Synthetic dataset
    x_train = torch.randn(128, 16)
    y_train = (x_train[:, 0] > 0).long()

    x_val = torch.randn(256, 16)
    y_val = (x_val[:, 0] > 0).long()

    model = Net()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()

    # Train model
    for _ in range(args.steps):
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation pass
    if args.fix:
        # FIX: Put model in eval mode for validation pass
        model.eval()
    else:
        # BUG: Model remains in train() mode during validation, corrupting test metrics
        pass

    with torch.no_grad():
        val_out = model(x_val)
        val_loss = float(criterion(val_out, y_val).item())

    print(f"Validation Loss: {val_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {val_loss}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
