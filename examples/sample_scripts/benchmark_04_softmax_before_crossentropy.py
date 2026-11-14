"""Benchmark 04: Softmax Applied Before CrossEntropyLoss.

Bug: Model explicitly applies softmax before CrossEntropyLoss, compressing gradients.
Taxonomy ID: softmax_before_cross_entropy
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 04: Softmax before CrossEntropyLoss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Output raw logits without softmax")
    parser.add_argument("--steps", type=int, default=60, help="Training steps")
    return parser.parse_args()


class Classifier(nn.Module):
    def __init__(self, use_softmax: bool):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 4)
        self.use_softmax = use_softmax

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        if self.use_softmax:
            # BUG: Applying softmax before CrossEntropyLoss
            return out
        # FIX: Output raw logits
        return out


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 10)
    y = torch.randint(0, 4, (64,))

    model = Classifier(use_softmax=not args.fix)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

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
