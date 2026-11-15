"""Benchmark 16: Vanishing Gradients Bug.

Bug: Stacking multiple Sigmoid activations in a deep network causes backpropagated gradients
to diminish exponentially (derivative bounded by 0.25), starving early layers of updates.
Taxonomy ID: vanishing_gradients
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 16: Vanishing Gradients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Use ReLU activations instead of Sigmoid")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    return parser.parse_args()


class DeepMLP(nn.Module):
    def __init__(self, use_relu: bool, depth: int = 8):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(16, 16))
            if use_relu:
                layers.append(nn.ReLU())
            else:
                # BUG: Saturated Sigmoid activations cause exponential gradient vanishing
                layers.append(nn.Sigmoid())
        layers.append(nn.Linear(16, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 16)
    w_true = torch.randn(16, 1)
    y = x @ w_true + 0.05 * torch.randn(64, 1)

    model = DeepMLP(use_relu=args.fix, depth=8)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
