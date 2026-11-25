"""Healthy training scripts, generated: the raw material the mutation operators break.

Each script trains on synthetic data drawn from a fixed "teacher", so it needs no
download and learns something real in about 60 optimizer steps on a CPU. The specs
vary what changes a training trace: the task, the architecture and its size, the
activation, dropout and batch norm, the optimizer and its learning rate, the batch
size, whether a DataLoader or manual slicing feeds the loop, and whether the model
is an `nn.Sequential` or a class with a `forward()`.

Version 4 adds what real notebooks do and the earlier generator never did: models
written as raw weight tensors updated by hand (no optimizer), GRU sequence models,
training and evaluation in functions, and healthy habits that look odd in a trace:
a learning-rate scheduler, gradient clipping, gradient accumulation over two
batches, and a summed loss paired with a learning rate divided by the batch size.
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
    # version 4
    loop: str = "inline"  # or "functions"
    scheduler: bool = False
    clip: bool = False
    accumulate: int = 1  # 2: step every second batch (with a DataLoader)
    healthy_sum: bool = False  # reduction="sum" with the learning rate divided by the batch size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def random_spec(rng: random.Random, data_seed: int = 0) -> Spec:
    task = rng.choice(["classification", "classification", "regression"])
    draw = rng.random()
    arch = "cnn" if task == "classification" and draw < 0.2 else ("rnn" if draw > 0.85 else "mlp")
    optimizer = rng.choice(list(OPTIMIZERS))
    low, high = OPTIMIZERS[optimizer][1]
    style = rng.choice(["sequential", "class"])
    if arch == "mlp" and rng.random() < 0.15:
        style = "manual"
    if arch == "rnn":
        style = "class"
    manual = style == "manual"
    loader = rng.random() < 0.6
    batchnorm = rng.random() < 0.3 and not manual and arch != "rnn"
    if manual:
        optimizer = "sgd"
        low, high = (0.05, 0.2) if task == "classification" else (0.01, 0.05)
    elif batchnorm and optimizer == "sgd" and rng.random() < 0.4:
        high = 0.8  # batch norm lets plain SGD take much larger steps
    spec = Spec(task=task, arch=arch, style=style, depth=rng.choice([1, 2, 3]),
                width=rng.choice([16, 32, 64, 128]), activation=rng.choice(list(ACTIVATIONS)),
                dropout=0.0 if manual or arch == "rnn" else rng.choice([0.0, 0.0, 0.1, 0.3]), batchnorm=batchnorm,
                optimizer=optimizer, lr=round(rng.uniform(low, high), 4), batch_size=rng.choice([16, 32, 64]),
                n_samples=rng.choice([256, 512, 1024]), n_features=rng.choice([8, 16, 32]),
                n_classes=rng.choice([2, 3, 5, 10]), loader=loader, data_seed=data_seed,
                loop="functions" if rng.random() < 0.3 else "inline",
                scheduler=not manual and rng.random() < 0.25, clip=rng.random() < 0.2,
                accumulate=2 if loader and not manual and rng.random() < 0.15 else 1)
    # a summed loss is healthy only when the step is made that much smaller; Adam ignores the scale,
    # but the project's taxonomy still calls sum-with-Adam a bug, so it is never generated as healthy
    spec.healthy_sum = not manual and optimizer in ("sgd", "momentum") and rng.random() < 0.15
    return spec


SEQUENCE_LENGTH = 8


def _data(spec: Spec) -> List[str]:
    n, d, c = spec.n_samples, spec.n_features, spec.n_classes
    shapes = {"cnn": (f"{n}, 1, 8, 8", f"{n // 4}, 1, 8, 8"),
              "rnn": (f"{n}, {SEQUENCE_LENGTH}, {d}", f"{n // 4}, {SEQUENCE_LENGTH}, {d}")}
    shape, val_shape = shapes.get(spec.arch, (f"{n}, {d}", f"{n // 4}, {d}"))
    inputs = 64 if spec.arch == "cnn" else d
    flat = "{x}.mean(dim=1)" if spec.arch == "rnn" else "{x}.reshape(len({x}), -1)"
    lines = [f"generator = torch.Generator().manual_seed({spec.data_seed})",
             f"x_train = torch.randn({shape}, generator=generator)",
             f"x_val = torch.randn({val_shape}, generator=generator)"]
    if spec.task == "classification":
        lines += [f"teacher = torch.randn({inputs}, {c}, generator=generator)",
                  f"y_train = ({flat.format(x='x_train')} @ teacher).argmax(dim=1)",
                  f"y_val = ({flat.format(x='x_val')} @ teacher).argmax(dim=1)"]
    else:
        scale = "* 3.0 " if spec.arch == "rnn" else ""
        lines += [f"teacher = torch.randn({inputs}, 1, generator=generator)",
                  f"y_train = {flat.format(x='x_train')} @ teacher {scale}+ 0.1 * torch.randn(len(x_train), 1, generator=generator)",
                  f"y_val = {flat.format(x='x_val')} @ teacher {scale}+ 0.1 * torch.randn(len(x_val), 1, generator=generator)"]
    return lines


def _manual_model(spec: Spec) -> List[str]:
    out = spec.n_classes if spec.task == "classification" else 1
    functional = ACTIVATIONS[spec.activation][1]
    sizes = [spec.n_features] + [spec.width] * spec.depth + [out]
    lines, names = [], []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        lines += [f"W{i} = ((torch.rand({a}, {b}) * 2 - 1) / math.sqrt({a})).requires_grad_()",
                  f"b{i} = torch.zeros({b}, requires_grad=True)"]
        names += [f"W{i}", f"b{i}"]
    lines += [f"params = [{', '.join(names)}]", "", "", "def model(x):"]
    last = len(sizes) - 2
    for i in range(last):
        lines.append(f"    x = {functional}(x @ W{i} + b{i})")
    return lines + [f"    return x @ W{last} + b{last}"]


def _model(spec: Spec) -> List[str]:
    out = spec.n_classes if spec.task == "classification" else 1
    module, functional = ACTIVATIONS[spec.activation]
    if spec.style == "manual":
        return _manual_model(spec)
    if spec.arch == "rnn":
        return ["class Net(nn.Module):", "    def __init__(self):", "        super().__init__()",
                f"        self.rnn = nn.GRU({spec.n_features}, {spec.width}, batch_first=True)",
                f"        self.head = nn.Linear({spec.width}, {out})", "", "    def forward(self, x):",
                "        out, _ = self.rnn(x)", "        return self.head(out[:, -1])", "", "model = Net()"]
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


def skips_mode_switch(spec: Spec) -> bool:
    """Without dropout or batch norm, train() and eval() change nothing, and real code often leaves them out."""
    return not spec.dropout and not spec.batchnorm and spec.data_seed % 2 == 0


def _batches(spec: Spec, indent: str) -> List[str]:
    """The batch loop header: a DataLoader or slicing by hand."""
    counter = spec.accumulate > 1
    if spec.loader:
        head = "for i, (xb, yb) in enumerate(train_loader):" if counter else "for xb, yb in train_loader:"
        return [indent + head]
    return [indent + "order = torch.randperm(len(x_train))",
            indent + f"for start in range(0, len(x_train), {spec.batch_size}):",
            indent + f"    batch = order[start:start + {spec.batch_size}]",
            indent + "    xb, yb = x_train[batch], y_train[batch]"]


def _step(spec: Spec, indent: str) -> List[str]:
    """One training step, healthy."""
    if spec.style == "manual":
        loss = "F.cross_entropy(model(xb), yb)" if spec.task == "classification" else "F.mse_loss(model(xb), yb)"
        clear = "p.grad.zero_()" if spec.data_seed % 2 else "p.grad = None"
        lines = [f"loss = {loss}", "loss.backward()"]
        if spec.clip:
            lines.append("torch.nn.utils.clip_grad_norm_(params, 1.0)")
        lines += ["with torch.no_grad():", "    for p in params:", "        p -= lr * p.grad", f"        {clear}"]
        return [indent + line for line in lines]
    clip = ["torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"] if spec.clip else []
    if spec.accumulate > 1:
        lines = ["output = model(xb)", f"loss = criterion(output, yb) / {spec.accumulate}", "loss.backward()",
                 f"if i % {spec.accumulate} == {spec.accumulate - 1}:", *["    " + c for c in clip],
                 "    optimizer.step()", "    optimizer.zero_grad()"]
    else:
        lines = ["optimizer.zero_grad()", "output = model(xb)", "loss = criterion(output, yb)", "loss.backward()",
                 *clip, "optimizer.step()"]
    return [indent + line for line in lines]


def _validation(spec: Spec, indent: str) -> List[str]:
    if spec.style == "manual":
        loss = "F.cross_entropy" if spec.task == "classification" else "F.mse_loss"
        return [indent + "with torch.no_grad():", indent + f"    val_loss = {loss}(model(x_val), y_val).item()"]
    return [*([] if skips_mode_switch(spec) else [indent + "model.eval()"]), indent + "with torch.no_grad():",
            indent + "    val_loss = criterion(model(x_val), y_val).item()"]


def _loop(spec: Spec) -> List[str]:
    manual = spec.style == "manual"
    train_mode = [] if skips_mode_switch(spec) or manual else ["model.train()"]
    lines = [f"train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size={spec.batch_size}, shuffle=True)"] \
        if spec.loader else []
    scheduler = ["    scheduler.step()"] if spec.scheduler else []
    report = "    print(f'epoch {epoch}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}')"
    if spec.loop == "functions":
        lines += ["", "", "def train_one_epoch():", *["    " + t for t in train_mode], *_batches(spec, "    "),
                  *_step(spec, "        "), "    return loss.item()", "", "", "def evaluate():",
                  *_validation(spec, "    "), "    return val_loss", "", "",
                  "for epoch in range(EPOCHS):", "    train_loss = train_one_epoch()", "    val_loss = evaluate()",
                  *scheduler, report]
        return lines
    return lines + ["for epoch in range(EPOCHS):", *["    " + t for t in train_mode], *_batches(spec, "    "),
                    *_step(spec, "        "), *_validation(spec, "    "), "    train_loss = loss.item()",
                    *scheduler, report]


def render(spec: Spec) -> str:
    """A complete, healthy training script for `spec`."""
    steps_per_epoch = -(-spec.n_samples // spec.batch_size)
    epochs = max(2, -(-80 * spec.accumulate // steps_per_epoch))
    reduction = '(reduction="sum")' if spec.healthy_sum else "()"
    criterion = ("nn.CrossEntropyLoss" if spec.task == "classification" else "nn.MSELoss") + reduction
    lr = spec.lr
    if spec.healthy_sum and spec.optimizer in ("sgd", "momentum"):
        lr = round(spec.lr / spec.batch_size, 6)  # a summed loss needs a step that many times smaller
    lines = ["import math", "import torch", "import torch.nn as nn", "import torch.nn.functional as F",
             "import torch.optim as optim", "from torch.utils.data import DataLoader, TensorDataset", "",
             f"EPOCHS = {epochs}", "torch.manual_seed(0)", ""]
    lines += _data(spec) + [""] + _model(spec) + [""]
    if spec.style == "manual":
        lines += [f"lr = {lr}", ""]
    else:
        lines += [f"criterion = {criterion}", "optimizer = " + OPTIMIZERS[spec.optimizer][0].format(lr=lr)]
        if spec.scheduler:
            lines.append("scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)")
        lines.append("")
    return "\n".join(lines + _loop(spec)) + "\n"
