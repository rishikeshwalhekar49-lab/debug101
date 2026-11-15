"""Benchmark 11: Duplicate Data Augmentation Applied Twice.

Bug: Data augmentation (transforms.RandomHorizontalFlip) is applied redundantly twice in the
transformation pipeline, introducing unexpected distribution distortion.
Taxonomy ID: duplicate_augmentation_applied_twice
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 11: Duplicate Augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Remove duplicate augmentation transform")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    if args.fix:
        # FIX: Single flip augmentation in pipeline
        pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        # BUG: Duplicate RandomHorizontalFlip applied twice in Compose pipeline
        pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    x = torch.randn(64, 1, 14, 14)
    y = torch.randint(0, 2, (64,))

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(14 * 14, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in range(args.steps):
        # Apply transformation pipeline to batch
        x_aug = pipeline(x)
        optimizer.zero_grad()
        pred = model(x_aug)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
