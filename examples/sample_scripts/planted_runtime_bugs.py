"""A plain training script with one optional planted silent bug (no monitoring code in it).

Run it under the Runtime Monitor Agent, which finds the model and optimizer by itself:

    python -m silent_bug_agent.agents.runtime_monitor_agent examples/sample_scripts/planted_runtime_bugs.py -- --bug frozen

Bugs: none, eval_mode, frozen, high_lr, dead_relu, partial_dead_relu, random_labels,
missing_zero_grad, nan_input, exploding. Same model and data as
examples/planted_bugs_torch.py, so the results can be compared.
"""

import argparse

import torch
import torch.nn as nn

BUGS = ["none", "eval_mode", "frozen", "high_lr", "dead_relu", "partial_dead_relu", "random_labels",
        "missing_zero_grad", "nan_input", "exploding"]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = self.drop(self.bn(self.act1(self.fc1(x))))
        return self.out(self.act2(self.fc2(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug", choices=BUGS, default="none")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    bug = args.bug

    torch.manual_seed(0)
    w_true = torch.randn(10, 3)
    model = Net()
    if bug == "dead_relu":
        with torch.no_grad():
            model.fc1.bias.fill_(-100.0)  # every unit of act1 is dead
    if bug == "partial_dead_relu":
        with torch.no_grad():
            model.fc1.bias[:20].fill_(-100.0)  # 20 of 32 units dead: zero_frac stays below 90%

    lr = 500.0 if bug == "high_lr" else 0.05
    params = [p for n, p in model.named_parameters() if not (bug == "frozen" and n.startswith("fc2"))]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for step in range(1, args.steps + 1):
        x = torch.randn(64, 10)
        y = (x @ w_true).argmax(dim=1)
        if bug == "random_labels":
            y = torch.randint(0, 3, (64,))
        if bug == "nan_input" and step >= 30:
            x[0, 0] = float("nan")  # a corrupt sample from step 30 on

        if bug != "missing_zero_grad":
            optimizer.zero_grad()
        loss = criterion(model(x), y)
        if bug == "exploding":
            loss = loss * 2.0 ** step  # a loss scale that doubles every step and is never unscaled
        loss.backward()
        optimizer.step()

        if step % 50 == 0:  # validation pass
            if bug != "eval_mode":
                model.eval()
            with torch.no_grad():
                val_loss = criterion(model(torch.randn(64, 10)), y)
            if bug != "eval_mode":
                model.train()
            print(f"step {step}: train loss {loss.item():.4f}  val loss {val_loss.item():.4f}")


if __name__ == "__main__":
    main()
