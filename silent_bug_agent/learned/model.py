"""The trace classifier: a tiny numpy model from training-trace features to bug probabilities.

It is either a softmax regression or a network with one tanh hidden layer, chosen
by `train.py`. Weights ship in `weights.json` (a few kilobytes, plain JSON), and
inference needs only numpy, which is already a dependency, so the package does not
grow and nothing is downloaded. One prediction takes microseconds.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .features import FEATURES, vectorize

WEIGHTS = Path(__file__).with_name("weights.json")
FORMAT = 1


class TraceClassifier:
    def __init__(self, classes: Sequence[str], features: Sequence[str], mean: Sequence[float],
                 scale: Sequence[float], layers: Sequence[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None):
        self.classes, self.features = list(classes), list(features)
        self.mean, self.scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
        self.layers = [(np.asarray(layer["W"], dtype=float), np.asarray(layer["b"], dtype=float)) for layer in layers]
        self.meta = dict(meta or {})

    # ------------------------------------------------------------ predict
    @property
    def temperature(self) -> float:
        """Calibration: logits are divided by this (fitted on the development scripts), so p means p."""
        return float(self.meta.get("temperature") or 1.0)

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = (np.nan_to_num(np.atleast_2d(np.asarray(X, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
             - self.mean) / self.scale
        for index, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if index < len(self.layers) - 1:
                h = np.tanh(h)
        return h

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = self.logits(X) / self.temperature
        h = h - h.max(axis=1, keepdims=True)
        p = np.exp(h)
        return p / p.sum(axis=1, keepdims=True)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.atleast_2d(np.asarray(X, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
        return self._forward(X)

    def predict_report(self, report: Dict[str, Any], source: Optional[str] = None) -> Dict[str, float]:
        """{label: probability} for one `script_runner --trace` report and its script, most likely first."""
        if self.features != FEATURES:
            raise ValueError("the saved classifier was trained on a different feature list; retrain it")
        p = self.probabilities(np.asarray([vectorize(report, source)]))[0]
        return dict(sorted(((c, round(float(v), 4)) for c, v in zip(self.classes, p)), key=lambda kv: -kv[1]))

    # --------------------------------------------------------------- fit
    @classmethod
    def fit(cls, X: np.ndarray, y: Sequence[str], *, hidden: int = 0, l2: float = 1e-3, epochs: int = 3000,
            lr: float = 0.03, seed: int = 0, classes: Optional[Sequence[str]] = None) -> "TraceClassifier":
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        classes = list(classes or sorted(set(y)))
        index = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((len(y), len(classes)))
        Y[np.arange(len(y)), [index[label] for label in y]] = 1.0
        counts = Y.sum(axis=0)
        weight = (Y @ (len(y) / (len(classes) * np.maximum(counts, 1.0))))[:, None]  # balanced classes
        mean, scale = X.mean(axis=0), X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        Z = (X - mean) / scale
        rng = np.random.default_rng(seed)
        sizes = [Z.shape[1]] + ([hidden] if hidden else []) + [len(classes)]
        params = []
        for a, b in zip(sizes, sizes[1:]):
            params += [rng.normal(0.0, 1.0 / np.sqrt(a), (a, b)), np.zeros(b)]
        moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        for step in range(1, epochs + 1):
            # forward
            activations = [Z]
            h = Z
            for layer in range(0, len(params), 2):
                h = h @ params[layer] + params[layer + 1]
                if layer < len(params) - 2:
                    h = np.tanh(h)
                activations.append(h)
            logits = activations[-1] - activations[-1].max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            # backward (weighted cross-entropy + L2)
            delta = weight * (P - Y) / len(y)
            grads: List[np.ndarray] = [np.zeros(0)] * len(params)
            for layer in range(len(params) - 2, -1, -2):
                inputs = activations[layer // 2]
                grads[layer] = inputs.T @ delta + l2 * params[layer]
                grads[layer + 1] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ params[layer].T) * (1.0 - inputs ** 2)
            for i, (p, g) in enumerate(zip(params, grads)):  # Adam
                m, v = moments[i]
                m[:] = 0.9 * m + 0.1 * g
                v[:] = 0.999 * v + 0.001 * g * g
                p -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        layers = [{"W": params[i], "b": params[i + 1]} for i in range(0, len(params), 2)]
        return cls(classes, FEATURES, mean, scale, layers, {"hidden": hidden, "l2": l2, "epochs": epochs})

    # ------------------------------------------------------------ storage
    def to_dict(self) -> Dict[str, Any]:
        def rounded(a: np.ndarray) -> Any:
            return np.round(a, 6).tolist()

        return {"format": FORMAT, "classes": self.classes, "features": self.features, "mean": rounded(self.mean),
                "scale": rounded(self.scale), "layers": [{"W": rounded(W), "b": rounded(b)} for W, b in self.layers],
                "meta": self.meta}

    def save(self, path: Path = WEIGHTS) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: Path = WEIGHTS) -> "TraceClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"unknown classifier format {data.get('format')!r}")
        return cls(data["classes"], data["features"], data["mean"], data["scale"], data["layers"], data.get("meta"))


@lru_cache(maxsize=1)
def load_default() -> Optional[TraceClassifier]:
    """The shipped classifier, or None if it is missing or was trained on other features."""
    try:
        model = TraceClassifier.load(WEIGHTS)
    except (OSError, ValueError, KeyError):
        return None
    return model if model.features == FEATURES else None
