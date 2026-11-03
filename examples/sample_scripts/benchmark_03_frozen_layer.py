"""Benchmark 03: Accidentally Frozen Layer.

Bug: Key layer omitted from optimizer param groups, preventing convergence.
Taxonomy ID: frozen_layer_no_grad
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 03: Frozen layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Include all model parameters in optimizer")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 8)
    y = torch.randn(64, 1)

    model = Model()

    if args.fix:
        # FIX: Train all parameters
        optimizer = optim.SGD(model.parameters(), lr=0.05)
    else:
        # BUG: Only fc1 is optimized; fc2 weights are frozen and never updated
        optimizer = optim.SGD(model.fc1.parameters(), lr=0.05)

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
