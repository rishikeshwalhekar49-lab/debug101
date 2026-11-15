"""Benchmark 12: Silent Device Fallback Bug.

Bug: Target device (GPU/CPU) is queried and constructed, but neither model nor data
tensors are ever placed onto the device, silently falling back to CPU default.
Taxonomy ID: silent_device_fallback
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 12: Silent Device Fallback")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fix", action="store_true", help="Explicitly move model and tensors to target device")
    parser.add_argument("--steps", type=int, default=40, help="Training steps")
    return parser.parse_args()


def run():
    args = parse_args()
    seed = int(os.environ.get("TORCH_SEED", args.seed))
    torch.manual_seed(seed)

    # Device initialized or queried
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(64, 10)
    w_true = torch.randn(10, 1)
    y = x @ w_true + 0.05 * torch.randn(64, 1)

    model = nn.Linear(10, 1)

    if args.fix:
        # FIX: Move model and batch tensors to the specified device
        place_on_device = getattr(model, "to")
        place_on_device(device)
        x = getattr(x, "to")(device)
        y = getattr(y, "to")(device)
    else:
        # BUG: Device is queried, but tensors and model remain silently on CPU
        pass

    optimizer = optim.SGD(model.parameters(), lr=0.05)
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
