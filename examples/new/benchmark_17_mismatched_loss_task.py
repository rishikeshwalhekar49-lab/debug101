"""Benchmark 17: Mismatched Loss for Task Bug.

Bug: Using Mean Squared Error (MSELoss) for a categorical classification problem,
treating discrete class labels as continuous regression targets.
Taxonomy ID: mismatched_loss_task
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 17: Mismatched Loss for Task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Use CrossEntropyLoss for multiclass classification")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, is_classification: bool):
        super().__init__()
        self.fc1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        # FIX outputs 4 categorical class logits; BUG outputs single continuous scalar
        self.fc2 = nn.Linear(32, 4 if is_classification else 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(128, 12)
    y = torch.randint(0, 4, (128,))

    model = Model(is_classification=args.fix)
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    if args.fix:
        # FIX: CrossEntropyLoss matches multiclass categorical objective
        criterion = nn.CrossEntropyLoss()
    else:
        # BUG: MSELoss incorrectly used on categorical classification targets
        criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(x)
        if args.fix:
            loss = criterion(pred, y)
        else:
            loss = criterion(pred.squeeze(-1), y.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
