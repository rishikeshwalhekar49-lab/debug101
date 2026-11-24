"""Healthy training scripts, generated: the raw material the mutation operators break.

Each script trains on synthetic data drawn from a fixed "teacher", so it needs no
download and learns something real in about 60 optimizer steps on a CPU. The specs
vary what changes a training trace: the task, the architecture and its size, the
activation, dropout and batch norm, the optimizer and its learning rate, the batch
size, whether a DataLoader or manual slicing feeds the loop, and whether the model
is an `nn.Sequential` or a class with a `forward()`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

ACTIVATIONS = {  # name -> (module, functional)
    "relu": ("nn.ReLU()", "F.relu"),
    "tanh": ("nn.Tanh()", "torch.tanh"),
    "gelu": ("nn.GELU()", "F.gelu"),
    "leaky_relu": ("nn.LeakyReLU(0.01)", "F.leaky_relu"),
}
OPTIMIZERS = {  # name -> (constructor, learning-rate range)
    "sgd": ("optim.SGD(model.parameters(), lr={lr})", (0.05, 0.2)),
    "momentum": ("optim.SGD(model.parameters(), lr={lr}, momentum=0.9)", (0.01, 0.05)),
    "adam": ("optim.Adam(model.parameters(), lr={lr})", (0.001, 0.005)),
    "adamw": ("optim.AdamW(model.parameters(), lr={lr}, weight_decay=0.01)", (0.001, 0.005)),
}


@dataclass
class Spec:
    task: str = "classification"  # or "regression"
    arch: str = "mlp"  # or "cnn" (classification only)
    style: str = "sequential"  # or "class"
    depth: int = 2
    width: int = 64
    activation: str = "relu"
    dropout: float = 0.0
    batchnorm: bool = False
    optimizer: str = "adam"
    lr: float = 0.003
    batch_size: int = 32
    n_samples: int = 512
    n_features: int = 16
    n_classes: int = 4
    loader: bool = True
    data_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def random_spec(rng: random.Random, data_seed: int = 0) -> Spec:
    task = rng.choice(["classification", "classification", "regression"])
    arch = "cnn" if task == "classification" and rng.random() < 0.25 else "mlp"
    optimizer = rng.choice(list(OPTIMIZERS))
    low, high = OPTIMIZERS[optimizer][1]
    return Spec(task=task, arch=arch, style=rng.choice(["sequential", "class"]), depth=rng.choice([1, 2, 3]),
                width=rng.choice([16, 32, 64, 128]), activation=rng.choice(list(ACTIVATIONS)),
                dropout=rng.choice([0.0, 0.0, 0.1, 0.3]), batchnorm=rng.random() < 0.3, optimizer=optimizer,
                lr=round(rng.uniform(low, high), 4), batch_size=rng.choice([16, 32, 64]),
                n_samples=rng.choice([256, 512, 1024]), n_features=rng.choice([8, 16, 32]),
                n_classes=rng.choice([2, 3, 5, 10]), loader=rng.random() < 0.6, data_seed=data_seed)


def _data(spec: Spec) -> List[str]:
    n, d, c = spec.n_samples, spec.n_features, spec.n_classes
    shape = f"{n}, 1, 8, 8" if spec.arch == "cnn" else f"{n}, {d}"
    val_shape = f"{n // 4}, 1, 8, 8" if spec.arch == "cnn" else f"{n // 4}, {d}"
    inputs = 64 if spec.arch == "cnn" else d
    lines = [f"generator = torch.Generator().manual_seed({spec.data_seed})",
             f"x_train = torch.randn({shape}, generator=generator)",
             f"x_val = torch.randn({val_shape}, generator=generator)"]
    if spec.task == "classification":
        lines += [f"teacher = torch.randn({inputs}, {c}, generator=generator)",
                  "y_train = (x_train.reshape(len(x_train), -1) @ teacher).argmax(dim=1)",
                  "y_val = (x_val.reshape(len(x_val), -1) @ teacher).argmax(dim=1)"]
    else:
        lines += [f"teacher = torch.randn({inputs}, 1, generator=generator)",
                  "y_train = x_train.reshape(len(x_train), -1) @ teacher + 0.1 * torch.randn(len(x_train), 1, generator=generator)",
                  "y_val = x_val.reshape(len(x_val), -1) @ teacher + 0.1 * torch.randn(len(x_val), 1, generator=generator)"]
    return lines


def _model(spec: Spec) -> List[str]:
    out = spec.n_classes if spec.task == "classification" else 1
    module, functional = ACTIVATIONS[spec.activation]
    if spec.arch == "cnn":
        if spec.style == "sequential":
            return ["model = nn.Sequential(", "    nn.Conv2d(1, 8, 3, padding=1),", f"    {module},",
                    "    nn.Flatten(),", f"    nn.Linear(8 * 8 * 8, {out}),", ")"]
        return ["class Net(nn.Module):", "    def __init__(self):", "        super().__init__()",
                "        self.conv = nn.Conv2d(1, 8, 3, padding=1)", f"        self.head = nn.Linear(8 * 8 * 8, {out})",
                "", "    def forward(self, x):", f"        x = {functional}(self.conv(x))",
                "        return self.head(x.flatten(1))", "", "model = Net()"]
    sizes = [spec.n_features] + [spec.width] * spec.depth
    if spec.style == "sequential":
        lines = ["model = nn.Sequential("]
        for a, b in zip(sizes, sizes[1:]):
            lines.append(f"    nn.Linear({a}, {b}),")
            if spec.batchnorm:
                lines.append(f"    nn.BatchNorm1d({b}),")
            lines.append(f"    {module},")
            if spec.dropout:
                lines.append(f"    nn.Dropout({spec.dropout}),")
        return lines + [f"    nn.Linear({sizes[-1]}, {out}),", ")"]
    lines = ["class Net(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        lines.append(f"        self.fc{i} = nn.Linear({a}, {b})")
        if spec.batchnorm:
            lines.append(f"        self.bn{i} = nn.BatchNorm1d({b})")
    if spec.dropout:
        lines.append(f"        self.drop = nn.Dropout({spec.dropout})")
    lines += [f"        self.out = nn.Linear({sizes[-1]}, {out})", "", "    def forward(self, x):"]
    for i in range(len(sizes) - 1):
        inner = f"self.bn{i}(self.fc{i}(x))" if spec.batchnorm else f"self.fc{i}(x)"
        lines.append(f"        x = {functional}({inner})")
        if spec.dropout:
            lines.append("        x = self.drop(x)")
    return lines + ["        return self.out(x)", "", "model = Net()"]


def _loop(spec: Spec) -> List[str]:
    if spec.loader:
        lines = [f"train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size={spec.batch_size}, shuffle=True)",
                 "for epoch in range(EPOCHS):", "    model.train()", "    for xb, yb in train_loader:"]
    else:
        lines = ["for epoch in range(EPOCHS):", "    model.train()", "    order = torch.randperm(len(x_train))",
                 f"    for start in range(0, len(x_train), {spec.batch_size}):",
                 f"        batch = order[start:start + {spec.batch_size}]",
                 "        xb, yb = x_train[batch], y_train[batch]"]
    return lines + ["        optimizer.zero_grad()", "        output = model(xb)", "        loss = criterion(output, yb)",
                    "        loss.backward()", "        optimizer.step()", "    model.eval()", "    with torch.no_grad():",
                    "        val_loss = criterion(model(x_val), y_val).item()",
                    "    print(f'epoch {epoch}: train loss {loss.item():.4f}, validation loss {val_loss:.4f}')"]


def render(spec: Spec) -> str:
    """A complete, healthy training script for `spec`."""
    steps_per_epoch = -(-spec.n_samples // spec.batch_size)
    epochs = max(2, -(-80 // steps_per_epoch))
    criterion = "nn.CrossEntropyLoss()" if spec.task == "classification" else "nn.MSELoss()"
    lines = ["import torch", "import torch.nn as nn", "import torch.nn.functional as F", "import torch.optim as optim",
             "from torch.utils.data import DataLoader, TensorDataset", "", f"EPOCHS = {epochs}", "torch.manual_seed(0)", ""]
    lines += _data(spec) + [""] + _model(spec) + [""]
    lines += [f"criterion = {criterion}", "optimizer = " + OPTIMIZERS[spec.optimizer][0].format(lr=spec.lr), ""]
    return "\n".join(lines + _loop(spec)) + "\n"
