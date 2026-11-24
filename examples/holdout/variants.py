"""The hand-written test set: 10 scripts, each healthy and with bugs planted by hand.

Written separately from the generator in `silent_bug_agent/mutation/`, in different
styles (training functions, a Trainer class, raw tensors with no nn.Module, an
EmbeddingBag text model, a GRU, an autoencoder, residual blocks, parameter groups,
schedulers, BCE and MSE losses), and each bug is planted by an exact text edit, often
in a different way from the mutation operators: zero_grad() called once before the
loop, a layer left out of the optimizer or out of its parameter groups, labels
reversed, gradients cleared by hand, train() mode during validation, and so on.

Scripts 01 to 04 are the development split: models may be tuned on them. Scripts 05
to 10 are the test split and are only ever used to report results.

Each edit is (text to find, replacement); the text must occur exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
Edit = Tuple[str, str]

VARIANTS: Dict[str, Dict] = {
    "holdout_01_tabular_functions.py": {
        "split": "dev",
        "healthy": [[("batch_size=64", "batch_size=128"), ("range(6)", "range(12)")]],
        "bugs": {
            "missing_zero_grad": [("        opt.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("    net.eval()\n", "    net.train()\n")],
            "softmax_before_cross_entropy": [("loss = loss_fn(net(xb), yb)",
                                              "loss = loss_fn(torch.softmax(net(xb), dim=1), yb)")],
            "wrong_loss_reduction": [("nn.CrossEntropyLoss()", 'nn.CrossEntropyLoss(reduction="sum")')],
            "learning_rate_misconfigured": [("lr=1e-3", "lr=1e1")],
            "frozen_layer_no_grad": [("torch.optim.Adam(net.parameters()", "torch.optim.Adam(net[4].parameters()")],
            "dead_relu": [("loss_fn = nn.CrossEntropyLoss()\n",
                           "nn.init.constant_(net[1].bias, -6.0)\nloss_fn = nn.CrossEntropyLoss()\n")],
            "non_normalized_inputs": [("features = centers[labels] + torch.randn(n, d, generator=g)",
                                       "features = (centers[labels] + torch.randn(n, d, generator=g)) * 200.0 + 500.0")],
            "label_data_misalignment": [("TensorDataset(x_tr, y_tr)", "TensorDataset(x_tr, y_tr[torch.randperm(len(y_tr))])")],
        },
    },
    "holdout_02_cnn_scheduler.py": {
        "split": "dev",
        "healthy": [[("batch = 32", "batch = 64"), ("lr=0.05", "lr=0.02")]],
        "bugs": {
            "missing_zero_grad": [("    optimizer.zero_grad()\n", "")],
            "softmax_before_cross_entropy": [("        return self.fc(x.flatten(1))",
                                              "        return F.softmax(self.fc(x.flatten(1)), dim=1)")],
            "wrong_loss_reduction": [("nn.CrossEntropyLoss()", "nn.CrossEntropyLoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=0.05", "lr=5.0")],
            "frozen_layer_no_grad": [("torch.optim.SGD(model.parameters()", "torch.optim.SGD(model.fc.parameters()")],
            "dead_relu": [("model = SmallCNN()\n", "model = SmallCNN()\nnn.init.constant_(model.conv1.bias, -10.0)\n")],
            "non_normalized_inputs": [("images = (images - mean) / std\nval_images = (val_images - mean) / std\n",
                                       "images = images * 255.0\nval_images = val_images * 255.0\n")],
            "label_data_misalignment": [("val_images, val_targets = synthetic_images(200, 4)\n",
                                         "val_images, val_targets = synthetic_images(200, 4)\n"
                                         "targets = targets[torch.randperm(len(targets))]\n")],
        },
    },
    "holdout_03_residual_regression.py": {
        "split": "dev",
        "healthy": [[("batch_size=50", "batch_size=100"), ("lr=2e-3", "lr=1e-3")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "wrong_loss_reduction": [("nn.MSELoss()", "nn.MSELoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=2e-3", "lr=2.0")],
            "frozen_layer_no_grad": [("torch.optim.AdamW(model.parameters()", "torch.optim.AdamW(model.head.parameters()")],
            "non_normalized_inputs": [("train_x, val_x = inputs[:1000], inputs[1000:]",
                                       "train_x, val_x = inputs[:1000] * 300.0 + 1000.0, inputs[1000:] * 300.0 + 1000.0")],
            "label_data_misalignment": [("train_y, val_y = targets[:1000], targets[1000:]",
                                         "train_y, val_y = targets[:1000][torch.randperm(1000)], targets[1000:]")],
        },
    },
    "holdout_04_text_bag.py": {
        "split": "dev",
        "healthy": [[("batch_size = 25", "batch_size = 50"), ("range(2)", "range(4)")]],
        "bugs": {
            "missing_zero_grad": [("        optim.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("    clf.eval()\n", "")],
            "softmax_before_cross_entropy": [("        return self.fc2(h)", "        return F.softmax(self.fc2(h), dim=-1)")],
            "wrong_loss_reduction": [("loss = F.cross_entropy(clf(train_tokens[rows]), train_labels[rows])",
                                      'loss = F.cross_entropy(clf(train_tokens[rows]), train_labels[rows], reduction="sum")')],
            "learning_rate_misconfigured": [("lr=5e-3", "lr=5e1")],
            "frozen_layer_no_grad": [("torch.optim.Adam(clf.parameters()", "torch.optim.Adam(clf.fc2.parameters()")],
            "dead_relu": [("clf = BagClassifier()\n", "clf = BagClassifier()\nnn.init.constant_(clf.fc1.bias, -8.0)\n")],
            "label_data_misalignment": [("train_tokens, train_labels = make_corpus(1500, 5)\n",
                                         "train_tokens, train_labels = make_corpus(1500, 5)\n"
                                         "train_labels = train_labels.flip(0)\n")],
        },
    },
    "holdout_05_trainer_class.py": {
        "split": "test",
        "healthy": [[("lr=3e-3, batch=48", "lr=1e-3, batch=96"), ("fit(epochs=5)", "fit(epochs=10)")]],
        "bugs": {
            "missing_zero_grad": [("                self.opt.zero_grad(set_to_none=True)\n", ""),
                                  ("        for epoch in range(epochs):\n",
                                   "        self.opt.zero_grad(set_to_none=True)\n        for epoch in range(epochs):\n")],
            "train_eval_mode_mismatch": [("        self.model.eval()\n", "")],
            "softmax_before_cross_entropy": [("        return self.out(self.body(x))",
                                              "        return torch.softmax(self.out(self.body(x)), dim=-1)")],
            "wrong_loss_reduction": [("nn.CrossEntropyLoss(label_smoothing=0.05)",
                                      "nn.CrossEntropyLoss(label_smoothing=0.05, reduction='sum')")],
            "learning_rate_misconfigured": [("lr=3e-3, batch=48", "lr=3.0, batch=48")],
            "frozen_layer_no_grad": [("        self.out = nn.Linear(width, classes)\n",
                                      "        self.out = nn.Linear(width, classes)\n"
                                      "        for p in self.body.parameters():\n            p.requires_grad = False\n")],
            "non_normalized_inputs": [("    y = (torch.tanh(x) @ w + 0.2 * torch.randn(n, 6, generator=g)).argmax(1)\n",
                                       "    y = (torch.tanh(x) @ w + 0.2 * torch.randn(n, 6, generator=g)).argmax(1)\n"
                                       "    x = x * 100.0 + 50.0\n")],
            "label_data_misalignment": [("x, y = dataset(960, 1)\n", "x, y = dataset(960, 1)\ny = y[torch.randperm(len(y))]\n")],
        },
    },
    "holdout_06_binary_logits.py": {
        "split": "test",
        "healthy": [[("batch_size = 20", "batch_size = 40"), ("lr=0.1", "lr=0.05")]],
        "bugs": {
            "missing_zero_grad": [("        sgd.zero_grad()\n", "")],
            "wrong_loss_reduction": [("nn.BCEWithLogitsLoss()", "nn.BCEWithLogitsLoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=0.1", "lr=30.0")],
            "frozen_layer_no_grad": [("torch.optim.SGD(model.parameters()", "torch.optim.SGD(model[4].parameters()")],
            "dead_relu": [("loss_function = nn.BCEWithLogitsLoss()\n",
                           "nn.init.constant_(model[0].bias, -9.0)\nloss_function = nn.BCEWithLogitsLoss()\n")],
            "non_normalized_inputs": [("points = torch.cat([points, points ** 2, points[:, :1] * points[:, 1:]], dim=1)\n",
                                       "points = torch.cat([points, points ** 2, points[:, :1] * points[:, 1:]], dim=1)"
                                       " * 150.0 + 400.0\n")],
            "label_data_misalignment": [("train_labels, test_labels = labels[:1600], labels[1600:]",
                                         "train_labels, test_labels = labels[:1600][torch.randperm(1600)], labels[1600:]")],
        },
    },
    "holdout_07_autoencoder.py": {
        "split": "test",
        "healthy": [[("lr=4e-3", "lr=2e-3"), ("], 40)", "], 80)")]],
        "bugs": {
            "missing_zero_grad": [("        adam.zero_grad()\n", "")],
            "wrong_loss_reduction": [("nn.MSELoss()", "nn.MSELoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=4e-3", "lr=4.0")],
            "frozen_layer_no_grad": [("torch.optim.Adam(ae.parameters()", "torch.optim.Adam(ae.decoder.parameters()")],
            "dead_relu": [("ae = AutoEncoder()\n", "ae = AutoEncoder()\nnn.init.constant_(ae.encoder[0].bias, -9.0)\n")],
            "non_normalized_inputs": [("train_signals, held_out = signals[:1200], signals[1200:]",
                                       "signals = signals * 255.0 + 255.0\n"
                                       "train_signals, held_out = signals[:1200], signals[1200:]")],
        },
    },
    "holdout_08_gru_sequences.py": {
        "split": "test",
        "healthy": [[("lr=3e-3", "lr=1e-3"), ("range(4)", "range(8)")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("    net.eval()\n", "")],
            "softmax_before_cross_entropy": [("        return self.head(self.dropout(hidden[-1]))",
                                              "        return torch.softmax(self.head(self.dropout(hidden[-1])), dim=1)")],
            "wrong_loss_reduction": [("nn.CrossEntropyLoss()", 'nn.CrossEntropyLoss(reduction="sum")')],
            "learning_rate_misconfigured": [("lr=3e-3", "lr=3e1")],
            "frozen_layer_no_grad": [("torch.optim.Adam(net.parameters()", "torch.optim.Adam(net.head.parameters()")],
            "non_normalized_inputs": [("val_seq, val_kind = sequences(256, 13)\n",
                                       "val_seq, val_kind = sequences(256, 13)\n"
                                       "train_seq, val_seq = train_seq * 100.0 + 40.0, val_seq * 100.0 + 40.0\n")],
            "label_data_misalignment": [("val_seq, val_kind = sequences(256, 13)\n",
                                         "val_seq, val_kind = sequences(256, 13)\n"
                                         "train_kind = train_kind[torch.randperm(len(train_kind))]\n")],
        },
    },
    "holdout_09_functional_tensors.py": {
        "split": "test",
        "healthy": [[("lr=0.2", "lr=0.1"), ("(32,)", "(64,)")]],
        "bugs": {
            "missing_zero_grad": [("    for p in params:\n        p.grad = None\n", "")],
            "softmax_before_cross_entropy": [("    loss = F.cross_entropy(forward(X[pick]), Y[pick])",
                                              "    loss = F.cross_entropy(F.softmax(forward(X[pick]), dim=1), Y[pick])")],
            "wrong_loss_reduction": [("    loss = F.cross_entropy(forward(X[pick]), Y[pick])",
                                      '    loss = F.cross_entropy(forward(X[pick]), Y[pick], reduction="sum")')],
            "learning_rate_misconfigured": [("lr=0.2", "lr=50.0")],
            "frozen_layer_no_grad": [("torch.optim.SGD(params, lr=0.2)", "torch.optim.SGD([w2, b2], lr=0.2)")],
            "dead_relu": [("b1 = torch.zeros(40, requires_grad=True)", "b1 = torch.full((40,), -6.0, requires_grad=True)")],
            "non_normalized_inputs": [("X_val, Y_val = X[800:], Y[800:]\n",
                                       "X = X * 400.0 + 900.0\nX_val, Y_val = X[800:], Y[800:]\n")],
            "label_data_misalignment": [("X, Y = X[:800], Y[:800]", "X, Y = X[:800], Y[:800][torch.randperm(800)]")],
        },
    },
    "holdout_10_param_groups.py": {
        "split": "test",
        "healthy": [[("batch_size=40", "batch_size=80"), ("range(4)", "range(8)")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("    network.eval()\n", "")],
            "softmax_before_cross_entropy": [("        return self.classifier(self.backbone(x))",
                                              "        return torch.softmax(self.classifier(self.backbone(x)), dim=1)")],
            "wrong_loss_reduction": [("nn.CrossEntropyLoss()", "nn.CrossEntropyLoss(reduction='sum')")],
            "learning_rate_misconfigured": [('"lr": 1e-3}', '"lr": 1e1}'), ('"lr": 3e-3}', '"lr": 3e1}')],
            "frozen_layer_no_grad": [('    {"params": network.backbone.parameters(), "lr": 1e-3},\n', "")],
            "dead_relu": [("network = Network()\n", "network = Network()\nnn.init.constant_(network.backbone[0].bias, -7.0)\n")],
            "non_normalized_inputs": [("dataset = TensorDataset(features[:1200], classes[:1200])",
                                       "features = features * 250.0 + 600.0\n"
                                       "dataset = TensorDataset(features[:1200], classes[:1200])")],
            "label_data_misalignment": [("dataset = TensorDataset(features[:1200], classes[:1200])",
                                         "dataset = TensorDataset(features[:1200], classes[:1200][torch.randperm(1200)])")],
        },
    },
}


def apply_edits(source: str, edits: List[Edit]) -> str:
    for find, replacement in edits:
        if source.count(find) != 1:
            raise ValueError(f"edit target must occur exactly once: {find!r}")
        source = source.replace(find, replacement)
    return source


def versions() -> List[Dict[str, str]]:
    """Every version: {"script", "split", "label", "code"}; healthy versions are labelled "clean"."""
    out = []
    for name, spec in VARIANTS.items():
        source = (HERE / name).read_text(encoding="utf-8")
        for edits in [[]] + spec["healthy"]:
            out.append({"script": name, "split": spec["split"], "label": "clean", "code": apply_edits(source, edits)})
        for bug, edits in spec["bugs"].items():
            out.append({"script": name, "split": spec["split"], "label": bug, "code": apply_edits(source, edits)})
    return out
