"""Benchmark 10: Gradient Accumulation Miscount Bug.

Bug: optimizer.zero_grad() is called unconditionally every step instead of inside the
gradient accumulation interval check, wiping out accumulated gradients prematurely.
Taxonomy ID: gradient_accumulation_miscount
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 10: Gradient Accumulation Miscount")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Zero gradients only after optimizer.step()")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def reset_grad(opt):
    opt.zero_grad()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    x = torch.randn(64, 10)
    w_true = torch.randn(10, 1)
    y = x @ w_true + 0.05 * torch.randn(64, 1)

    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    accum_steps = 4
    losses = []
    for step in range(args.steps):
        pred = model(x)
        loss = criterion(pred, y) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            if args.fix:
                # FIX: Only zero gradients after optimizer steps across accumulated batches
                reset_grad(optimizer)

        if not args.fix:
            # BUG: Calling zero_grad() unconditionally outside the accumulation block
            optimizer.zero_grad()

        losses.append(float(loss.item() * accum_steps))

    final_loss = losses[-1]
    print(f"Final Loss: {final_loss:.6f}")
    print(f'__METRICS__:{{"final_loss": {final_loss}, "initial_loss": {losses[0]}, "steps": {args.steps}}}')


if __name__ == "__main__":
    run()
