"""Training scripts from Google Colab notebooks, healthy and with bugs planted.

Each `colab_*.py` follows a notebook that runs in Colab (the PyTorch tutorials and
d2l.ai; see README.md here) and, where the notebook downloads a dataset, trains on
Colab's own `/content/sample_data` files instead (MNIST and California housing).
`fetch_sample_data.py` puts copies of those files in a local cache, and `versions()`
points the scripts there.

Three splits, by script, never by version:

* "train": the classifier learns from these. Their bugs are planted by the mutation
  operators (silent_bug_agent/mutation/operators.py), like the generated scripts.
* "dev": settings are chosen on these, and only their mistakes are studied.
* "test": only ever used to report results. Their bugs are planted by hand-written
  edits, often in ways the operators never use (labels sorted or rolled by one,
  a custom loss that sums, only some weights updated by hand, a one-hot input of
  255, the BatchNorm net left in training mode for its test batch).

Each edit is (text to find, replacement); the text must occur exactly once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
COLAB_PATH = '"/content/sample_data"'
Edit = Tuple[str, str]
FILES = ("mnist_train_small.csv", "mnist_test.csv", "california_housing_train.csv", "california_housing_test.csv")


def sample_data_dir() -> Path:
    return Path(os.environ.get("SBA_SAMPLE_DATA", Path.home() / ".cache" / "silent_bug_agent" / "sample_data"))


def has_sample_data(folder: Optional[Path] = None) -> bool:
    folder = folder or sample_data_dir()
    return all((folder / name).is_file() for name in FILES)


TRAIN = ["colab_01_quickstart_mnist.py", "colab_04_nn_tutorial.py", "colab_06_housing_regression.py",
         "colab_08_lstm_tagger.py", "colab_12_d2l_linear_regression_scratch.py", "colab_13_d2l_mlp_concise.py",
         "colab_16_autoencoder_mnist.py"]
# a second healthy version of each training script: slower, so a slow healthy run is not taken for a bug
TRAIN_HEALTHY: Dict[str, List[Edit]] = {
    "colab_01_quickstart_mnist.py": [("learning_rate = 1e-2", "learning_rate = 5e-3")],
    "colab_04_nn_tutorial.py": [("lr = 0.1", "lr = 0.05")],
    "colab_06_housing_regression.py": [("lr=0.003", "lr=0.001")],
    "colab_08_lstm_tagger.py": [("lr=0.1", "lr=0.05")],
    "colab_12_d2l_linear_regression_scratch.py": [("lr = 0.03", "lr = 0.01")],
    "colab_13_d2l_mlp_concise.py": [("batch_size, lr, num_epochs = 64, 0.1, 10", "batch_size, lr, num_epochs = 128, 0.05, 10")],
    "colab_16_autoencoder_mnist.py": [("batch_size=32", "batch_size=64")],
}

VARIANTS: Dict[str, Dict] = {
    # ------------------------------------------------------------------ dev
    "colab_02_training_with_pytorch.py": {
        "split": "dev",
        "healthy": [[("batch_size=16, shuffle=True", "batch_size=32, shuffle=True")],
                    [("lr=0.01, momentum=0.9", "lr=0.005, momentum=0.9")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "softmax_before_cross_entropy": [("        x = self.fc3(x)\n        return x",
                                              "        x = F.softmax(self.fc3(x), dim=1)\n        return x")],
            "wrong_loss_reduction": [("loss_fn = torch.nn.CrossEntropyLoss()",
                                      "loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=0.01, momentum=0.9", "lr=0.9, momentum=0.9")],
            "frozen_layer_no_grad": [("model = GarmentClassifier()\n",
                                      "model = GarmentClassifier()\nfor param in model.conv1.parameters():\n"
                                      "    param.requires_grad = False\nfor param in model.conv2.parameters():\n"
                                      "    param.requires_grad = False\n")],
            "dead_relu": [("model = GarmentClassifier()\n",
                           "model = GarmentClassifier()\nnn.init.constant_(model.fc1.bias, -4.0)\n")],
            "non_normalized_inputs": [("(torch.from_numpy(data[:, 1:]).reshape(-1, 1, 28, 28) / 255.0 - 0.5) / 0.5",
                                       "torch.from_numpy(data[:, 1:]).reshape(-1, 1, 28, 28)")],
            "label_data_misalignment": [("    return TensorDataset(images, torch.from_numpy(data[:, 0]).long())",
                                         "    return TensorDataset(images, torch.from_numpy(np.roll(data[:, 0], 1)).long())")],
        },
    },
    "colab_07_polynomial_fit.py": {
        "split": "dev",
        "healthy": [[("learning_rate = 1e-6", "learning_rate = 5e-7")],
                    [("x = torch.linspace(-math.pi, math.pi, 2000)", "x = torch.linspace(-math.pi, math.pi, 1000)")]],
        "bugs": {
            "missing_zero_grad": [("    model.zero_grad()\n", "")],
            "learning_rate_misconfigured": [("learning_rate = 1e-6", "learning_rate = 1e-4")],
            "non_normalized_inputs": [("xx = x.unsqueeze(-1).pow(p)", "xx = (x * 100).unsqueeze(-1).pow(p)")],
            "label_data_misalignment": [("y = torch.sin(x)", "y = torch.sin(x)[torch.randperm(2000)]")],
        },
    },
    "colab_11_text_classification.py": {
        "split": "dev",
        "healthy": [[("BATCH_SIZE = 64", "BATCH_SIZE = 32")], [("LR = 5", "LR = 2")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n        predicted_label = model(text, offsets)",
                                   "        predicted_label = model(text, offsets)")],
            "softmax_before_cross_entropy": [("        return self.fc(embedded)",
                                              "        return torch.softmax(self.fc(embedded), dim=1)")],
            "wrong_loss_reduction": [("criterion = torch.nn.CrossEntropyLoss()",
                                      'criterion = torch.nn.CrossEntropyLoss(reduction="sum")')],
            "learning_rate_misconfigured": [("LR = 5", "LR = 0.005")],
            "frozen_layer_no_grad": [("optimizer = torch.optim.SGD(model.parameters(), lr=LR)",
                                      "optimizer = torch.optim.SGD(model.fc.parameters(), lr=LR)")],
            "label_data_misalignment": [("        label_list.append(_label)",
                                         "        label_list.append(int(torch.randint(0, NUM_CLASS, (1,))))")],
        },
    },
    "colab_17_housing_binary.py": {
        "split": "dev",
        "healthy": [[("batch_size = 128", "batch_size = 256")], [("lr=0.01", "lr=0.003")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "wrong_loss_reduction": [("loss_fn = nn.BCEWithLogitsLoss()", "loss_fn = nn.BCEWithLogitsLoss(reduction='sum')")],
            "learning_rate_misconfigured": [("lr=0.01", "lr=10.0")],
            "frozen_layer_no_grad": [("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)",
                                      "optimizer = torch.optim.Adam(model.output.parameters(), lr=0.01)")],
            "dead_relu": [("model = Classifier(len(columns))\n",
                           "model = Classifier(len(columns))\nnn.init.constant_(model.hidden.bias, -10.0)\n")],
            "non_normalized_inputs": [("X_train = torch.tensor(train_norm.values, dtype=torch.float32)",
                                       "X_train = torch.tensor(train_df[columns].values, dtype=torch.float32)"),
                                      ("X_test = torch.tensor(test_norm.values, dtype=torch.float32)",
                                       "X_test = torch.tensor(test_df[columns].values, dtype=torch.float32)")],
            "label_data_misalignment": [('y_train = torch.tensor(train_df["expensive"].values).unsqueeze(1)',
                                         'y_train = torch.tensor(train_df["expensive"].sample(frac=1.0).values).unsqueeze(1)')],
        },
    },
    # ----------------------------------------------------------------- test
    "colab_03_cifar_tutorial_net.py": {
        "split": "test",
        "healthy": [[("batch_size=8, shuffle=True", "batch_size=16, shuffle=True")],
                    [("lr=0.01, momentum=0.9", "lr=0.005, momentum=0.9")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "softmax_before_cross_entropy": [("        x = self.fc3(x)\n        return x",
                                              "        x = self.fc3(x)\n        return F.softmax(x, dim=1)")],
            "wrong_loss_reduction": [("criterion = nn.CrossEntropyLoss()", 'criterion = nn.CrossEntropyLoss(reduction="sum")')],
            "learning_rate_misconfigured": [("lr=0.01, momentum=0.9", "lr=1e-5, momentum=0.9")],
            "frozen_layer_no_grad": [("optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)",
                                      "optimizer = optim.SGD(net.fc3.parameters(), lr=0.01, momentum=0.9)")],
            "dead_relu": [("net = Net()\n", "net = Net()\nwith torch.no_grad():\n    net.conv1.bias.fill_(-5.0)\n")],
            "non_normalized_inputs": [("reshape(-1, 1, 28, 28) / 255.0\n", "reshape(-1, 1, 28, 28)\n")],
            "label_data_misalignment": [("targets = torch.from_numpy(raw[:, 0]).long()",
                                         "targets = torch.from_numpy(np.sort(raw[:, 0])).long()")],
        },
    },
    "colab_05_nn_tutorial_from_scratch.py": {
        "split": "test",
        "healthy": [[("bs = 64", "bs = 128")], [("lr = 0.5", "lr = 0.2")]],
        "bugs": {
            "missing_zero_grad": [("            weights.grad.zero_()\n            bias.grad.zero_()\n", "")],
            "learning_rate_misconfigured": [("lr = 0.5", "lr = 50.0")],
            "wrong_loss_reduction": [("    return -input[range(target.shape[0]), target].mean()",
                                      "    return -input[range(target.shape[0]), target].sum()")],
            "frozen_layer_no_grad": [("            weights -= weights.grad * lr\n", "")],
            "non_normalized_inputs": [("map(torch.tensor, (train[:, 1:] / 255.0, train[:, 0], valid[:, 1:] / 255.0,",
                                       "map(torch.tensor, (train[:, 1:], train[:, 0], valid[:, 1:],")],
            "label_data_misalignment": [("y_train, y_valid = y_train.long(), y_valid.long()",
                                         "y_train, y_valid = y_train[torch.randperm(len(y_train))].long(), y_valid.long()")],
        },
    },
    "colab_09_ngram_language_model.py": {
        "split": "test",
        "healthy": [[("lr=0.05", "lr=0.02")], [("EMBEDDING_DIM = 10", "EMBEDDING_DIM = 16")]],
        "bugs": {
            "missing_zero_grad": [("        model.zero_grad()\n", "")],
            "learning_rate_misconfigured": [("lr=0.05", "lr=50.0")],
            "frozen_layer_no_grad": [("optimizer = optim.SGD(model.parameters(), lr=0.05)",
                                      "optimizer = optim.SGD(model.linear2.parameters(), lr=0.05)")],
            "dead_relu": [("model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)\n",
                           "model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)\n"
                           "nn.init.constant_(model.linear1.bias, -8.0)\n")],
            "label_data_misalignment": [("import torch\n", "import random\n\nimport torch\n"),
                                        ("    ([test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], test_sentence[i])",
                                         "    ([test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], random.choice(test_sentence))")],
        },
    },
    "colab_10_char_rnn_names.py": {
        "split": "test",
        "healthy": [[("learning_rate = 0.005", "learning_rate = 0.003")], [("n_hidden = 128", "n_hidden = 64")]],
        "bugs": {
            "missing_zero_grad": [("    rnn.zero_grad()\n", "")],
            "learning_rate_misconfigured": [("learning_rate = 0.005", "learning_rate = 1.0")],
            "frozen_layer_no_grad": [("    for p in rnn.parameters():\n        p.data.add_(p.grad.data, alpha=-learning_rate)",
                                      "    for p in rnn.h2o.parameters():\n        p.data.add_(p.grad.data, alpha=-learning_rate)")],
            "non_normalized_inputs": [("        tensor[li][0][letterToIndex(letter)] = 1",
                                       "        tensor[li][0][letterToIndex(letter)] = 255")],
            "label_data_misalignment": [("    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)",
                                         "    category_tensor = torch.tensor([random.randrange(n_categories)], dtype=torch.long)")],
        },
    },
    "colab_14_d2l_dropout.py": {
        "split": "test",
        "healthy": [[("num_epochs, lr, batch_size = 10, 0.5, 256", "num_epochs, lr, batch_size = 10, 0.3, 256")],
                    [("dropout1, dropout2 = 0.2, 0.5", "dropout1, dropout2 = 0.1, 0.3")]],
        "bugs": {
            "missing_zero_grad": [("        trainer.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("    net.eval()\n", "")],
            "softmax_before_cross_entropy": [("                    nn.Linear(256, 10))",
                                              "                    nn.Linear(256, 10), nn.Softmax(dim=1))")],
            "wrong_loss_reduction": [("loss = nn.CrossEntropyLoss()", "loss = nn.CrossEntropyLoss(reduction='sum')")],
            "learning_rate_misconfigured": [("num_epochs, lr, batch_size = 10, 0.5, 256",
                                             "num_epochs, lr, batch_size = 10, 0.0005, 256")],
            "frozen_layer_no_grad": [("trainer = torch.optim.SGD(net.parameters(), lr=lr)",
                                      "trainer = torch.optim.SGD(net[4].parameters(), lr=lr)")],
            "dead_relu": [("        nn.init.normal_(m.weight, std=0.01)\n",
                           "        nn.init.normal_(m.weight, std=0.01)\n        nn.init.constant_(m.bias, -1.0)\n")],
            "non_normalized_inputs": [("torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28) / 255,",
                                       "torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28),")],
            "label_data_misalignment": [("                                          torch.from_numpy(a[:, 0]).long())",
                                         "                                          torch.from_numpy(a[:, 0]).long()[torch.randperm(len(a))])")],
        },
    },
    "colab_15_d2l_batchnorm_lenet.py": {
        "split": "test",
        "healthy": [[("lr, num_epochs, batch_size = 1.0, 10, 128", "lr, num_epochs, batch_size = 0.5, 10, 128")],
                    [("lr, num_epochs, batch_size = 1.0, 10, 128", "lr, num_epochs, batch_size = 1.0, 10, 256")]],
        "bugs": {
            "missing_zero_grad": [("        optimizer.zero_grad()\n", "")],
            "train_eval_mode_mismatch": [("            net.eval()\n", "")],
            "softmax_before_cross_entropy": [("    nn.Linear(84, 10))", "    nn.Linear(84, 10), nn.Softmax(dim=1))")],
            "wrong_loss_reduction": [("loss = nn.CrossEntropyLoss()", 'loss = nn.CrossEntropyLoss(reduction="sum")')],
            "learning_rate_misconfigured": [("lr, num_epochs, batch_size = 1.0, 10, 128",
                                             "lr, num_epochs, batch_size = 200.0, 10, 128")],
            "frozen_layer_no_grad": [("optimizer = torch.optim.SGD(net.parameters(), lr=lr)",
                                      "optimizer = torch.optim.SGD(list(net.parameters())[-6:], lr=lr)")],
            "non_normalized_inputs": [("torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28) / 255,",
                                       "torch.from_numpy(a[:, 1:]).reshape(-1, 1, 28, 28),")],
            "label_data_misalignment": [("train_iter = data.DataLoader(to_set(train), batch_size, shuffle=True)",
                                         "train[:, 0] = np.random.permutation(train[:, 0])\n"
                                         "train_iter = data.DataLoader(to_set(train), batch_size, shuffle=True)")],
        },
    },
}


def apply_edits(source: str, edits: List[Edit]) -> str:
    for find, replacement in edits:
        if source.count(find) != 1:
            raise ValueError(f"edit target must occur exactly once: {find!r}")
        source = source.replace(find, replacement)
    return source


def localise(source: str, folder: Optional[Path] = None) -> str:
    """Point a script at the local copy of Colab's sample_data folder."""
    return source.replace(COLAB_PATH, f'"{(folder or sample_data_dir()).as_posix()}"')


def versions(folder: Optional[Path] = None) -> List[Dict[str, str]]:
    """Every version: {"script", "split", "label", "code"}; healthy versions are labelled "clean".

    Training scripts come as their healthy versions only (the caller plants bugs with
    the operators). Scripts that need the sample data are left out when it is missing.
    """
    folder = folder or sample_data_dir()
    available = has_sample_data(folder)
    out = []
    for name in TRAIN:
        source = (HERE / name).read_text(encoding="utf-8")
        if COLAB_PATH in source and not available:
            continue
        for edits in ([], TRAIN_HEALTHY.get(name)):
            if edits is not None:
                out.append({"script": name, "split": "train", "label": "clean",
                            "code": localise(apply_edits(source, edits), folder)})
    for name, spec in VARIANTS.items():
        source = (HERE / name).read_text(encoding="utf-8")
        if COLAB_PATH in source and not available:
            continue
        for edits in [[]] + spec["healthy"]:
            out.append({"script": name, "split": spec["split"], "label": "clean",
                        "code": localise(apply_edits(source, edits), folder)})
        for bug, edits in spec["bugs"].items():
            out.append({"script": name, "split": spec["split"], "label": bug,
                        "code": localise(apply_edits(source, edits), folder)})
    return out
