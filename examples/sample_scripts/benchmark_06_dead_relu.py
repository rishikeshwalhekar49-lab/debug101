"""Benchmark 06: Dead ReLU Units.

Bug: Large negative bias initialization causes 100% dead ReLU neurons, blocking gradients.
Taxonomy ID: dead_relu
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 06: Dead ReLU units")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Fix bias initialization to zero")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, dead_relu: bool):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        if dead_relu:
            # BUG: Large negative bias pushes all pre-activations negative
            with torch.no_grad():
                self.fc1.bias.fill_(-100.0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 8)
    y = torch.randn(64, 1)

    model = Model(dead_relu=not args.fix)
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
