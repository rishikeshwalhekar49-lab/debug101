"""Benchmark 15: Exploding Gradients Bug.

Bug: Deep network layers with poor initialization cause gradient norms to explode during
backpropagation, causing instability unless gradient clipping is applied.
Taxonomy ID: exploding_gradients
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 15: Exploding Gradients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Apply gradient clipping to prevent explosion")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


class DeepLinear(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        layers = []
        for _ in range(depth):
            l = nn.Linear(16, 16)
            # High standard deviation on weights to induce exponential gradient compounding
            nn.init.normal_(l.weight, mean=0.0, std=1.8)
            layers.append(l)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 16)
    y = torch.randn(64, 1)

    model = DeepLinear(depth=6)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        if args.fix:
            # FIX: Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            # BUG: Unclipped gradients explode through deep unnormalized linear blocks
            pass

        optimizer.step()
        val = float(loss.item())
        if not torch.isfinite(loss):
            val = 999.0
        losses.append(min(val, 999.0))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
