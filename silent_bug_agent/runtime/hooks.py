"""Module 1: non-intrusive PyTorch hooks that feed the Runtime Monitor Agent.

``RuntimeHooks`` attaches to an existing model and optimizer -- the user's
model source is never modified -- and turns training into a stream of
:class:`~silent_bug_agent.invariants.base.TrainingState` snapshots, the same
framework-agnostic contract the Module 0 invariants already check.

Sensors (all removable with :meth:`RuntimeHooks.detach`):

* **forward hooks** on every leaf layer (plus ``nn.MultiheadAttention`` and RNN
  modules as units) -> ``activation_stats``: the contract keys ``mean``, ``std``,
  ``nan_count``, ``inf_count``, ``zero_frac`` plus ``min``, ``max``, ``abs_mean``,
  ``saturated_frac`` (tanh/sigmoid), ``dead_unit_frac`` (ReLU layers, batches of at
  least 16: fraction of units that output zero for every sample in the batch),
  ``norm``, ``numel``, ``calls``;
* **backward (tensor) hooks** on each layer's output -> gradient flowing back
  through the layer: ``grad_norm``, ``grad_abs_mean``, ``grad_abs_max``,
  ``grad_nan_count``, ``grad_inf_count`` inside the same ``activation_stats`` entry;
* **optimizer step pre/post hooks** -> ``grad_norms`` and ``weight_deltas`` per
  parameter (``||W_after - W_before||`` of that step), ``requires_grad``;
* a **backward() wrapper** -> ``loss_history`` plus loss dynamics (EMA, z-score,
  spike flag) in ``extra["loss"]``;
* a **root forward hook** -> automatic ``is_eval_phase=True`` snapshots when the
  model is run under ``torch.no_grad()`` (validation), with ``module_modes``.

Labels follow ``examples/torch_adapter.py``: layers and modules are keyed
``"<name>[<ClassName>]"`` so the mode invariant can match "batchnorm"/"dropout";
parameters are keyed by their ``named_parameters()`` name.

Low-overhead streaming (adapted from package101 ``trainmedic/detect/tracer.py``):

* statistics are computed on the tensor's own device (GPU or CPU). A hook only
  copies a strided sample of its tensor (at most ``max_elems``, default 16,384
  elements) into its row of a staging buffer allocated once; when the tensor is
  larger, one extra reduction over the whole tensor still catches any NaN/Inf
  outside the sample. At the end of the step the statistics of *all* layers are
  computed together in a few dozen batched kernels, and parameter norms use fused
  multi-tensor kernels: on a GPU, kernel launches, not arithmetic, are the cost;
* hooks never synchronise with the GPU. At the end of a monitored step every
  statistic is packed into one small device vector, copied to the host once and
  asynchronously into a reused pinned buffer, and turned into a TrainingState
  when the copy has landed (at most ``max_pending`` steps later);
* the only extra device copy is a strided sample of each weight tensor, taken
  just before ``optimizer.step()`` into a buffer allocated once, so the weight
  delta can be measured;
* snapshots go to a :class:`SnapshotStream` (bounded, thread-safe, never blocks
  training) for the Runtime Monitor Agent to consume.

Validation across architectures: ``python -m silent_bug_agent.runtime.validate``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..invariants.base import Invariant, InvariantResult, TrainingState
from ..invariants.registry import run_all
from .numerics import EmaZScore

ACT_FIELDS = ("mean", "std", "min", "max", "abs_mean", "zero_frac", "saturated_frac", "norm", "nan_count", "inf_count",
              "dead_unit_frac")
GRAD_FIELDS = ("grad_norm", "grad_abs_mean", "grad_abs_max", "grad_nan_count", "grad_inf_count")
PARAM_FIELDS = ("grad_norm", "grad_abs_max", "weight_norm", "delta_norm")
# dead_unit_frac needs this many samples: "zero for every sample" of a handful is weak evidence of a dead unit
MIN_DEAD_UNIT_BATCH = 16

_SKIP_TYPES: Tuple[type, ...] = (nn.Identity, nn.Flatten, nn.Unflatten, nn.modules.loss._Loss)
_UNIT_TYPES: Tuple[type, ...] = (nn.MultiheadAttention, nn.RNNBase)


def _compiling() -> bool:
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if fn is None:
        try:
            from torch._dynamo import is_compiling as fn  # torch 2.0
        except Exception:
            return False
    try:
        return bool(fn())
    except Exception:
        return False


def module_label(name: str, module: nn.Module) -> str:
    """``"<name>[<ClassName>]"`` -- the key convention of examples/torch_adapter.py."""
    return f"{name}[{type(module).__name__}]"


def _saturation_kind(module: nn.Module) -> str:
    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        return "relu"  # no saturation, but units can die: dead_unit_frac is measured
    if isinstance(module, (nn.Tanh, nn.Hardtanh)):
        return "tanh"
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    return ""


def _first_float_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj if obj.is_floating_point() and obj.numel() > 0 else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_float_tensor(o)
            if t is not None:
                return t
    return None


# =============================================================== statistics
class TensorStatsExtractor:
    """Tensor statistics computed on the tensor's own device.

    Every method returns a small device tensor and never calls ``.item()``,
    ``float()`` or ``.tolist()``, so a hook never forces a GPU synchronisation."""

    def __init__(self, max_elems: int = 1 << 14, histogram_bins: int = 0) -> None:
        self.max_elems = max_elems
        self.histogram_bins = histogram_bins

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """A flat strided *view* of at most ``max_elems`` elements, and its stride."""
        flat = x.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        stride = max(1, math.ceil(flat.numel() / self.max_elems))
        return (flat[::stride] if stride > 1 else flat), stride

    @staticmethod
    def _f32(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.float32 else t.float()

    @staticmethod
    def _nan_inf(x: torch.Tensor, f: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nan = torch.isnan(f).sum().to(torch.float32) * stride
        inf = torch.isinf(f).sum().to(torch.float32) * stride
        if stride > 1:
            # One extra reduction over the whole tensor: NaN/Inf anywhere propagate into the sum, so a
            # bad value outside the sample still gives a count >= 1. (If NaN *and* Inf both occur only
            # outside the sample, the sum is NaN and only nan_count is raised.)
            total = x.detach().sum(dtype=torch.float32)
            one = torch.ones((), dtype=torch.float32, device=f.device)
            nan = torch.where(torch.isnan(total) & (nan == 0), one, nan)
            inf = torch.where(torch.isinf(total) & (inf == 0), one, inf)
        return nan, inf

    def activation(self, x: torch.Tensor, saturation: str = "") -> torch.Tensor:
        f, stride = self.sample(x)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        nf = n.clamp_min(1.0)
        zero = torch.zeros((), dtype=f.dtype, device=f.device)
        z = torch.where(finite, f, zero)
        s2 = (z * z).sum()
        mean = z.sum() / nf
        std = (s2 / nf - mean * mean).clamp_min(0).sqrt()
        inf_t = torch.full((), float("inf"), dtype=f.dtype, device=f.device)
        mn = torch.where(finite, f, inf_t).amin()
        mx = torch.where(finite, f, -inf_t).amax()
        abs_mean = z.abs().sum() / nf
        zero_frac = ((f == 0) & finite).sum().to(f.dtype) / nf
        if saturation == "tanh":
            sat = (z.abs() >= 0.99).sum().to(f.dtype) / nf
        elif saturation == "sigmoid":
            sat = (((z <= 0.01) | (z >= 0.99)) & finite).sum().to(f.dtype) / nf
        else:
            sat = zero
        nan, inf = self._nan_inf(x, f, stride)
        dead = self._dead_units(x) if saturation == "relu" else torch.full((), float("nan"), dtype=f.dtype, device=f.device)
        return torch.stack([mean, std, mn, mx, abs_mean, zero_frac, sat, s2.sqrt() * math.sqrt(stride), nan, inf, dead])

    def _dead_units(self, x: torch.Tensor) -> torch.Tensor:
        """Fraction of units (all non-batch positions) that are zero for every sample of the batch.

        A dying-ReLU layer is often only partly dead, so its overall ``zero_frac`` stays
        below the 90% activation_health threshold while many units never fire. Units are
        sampled with a strided view (columns) like the other statistics."""
        if x.dim() < 2 or x.shape[0] < MIN_DEAD_UNIT_BATCH:
            return torch.full((), float("nan"), dtype=torch.float32, device=x.device)
        u = x.detach().reshape(x.shape[0], -1)
        stride = max(1, math.ceil(u.numel() / self.max_elems))
        if stride > 1:
            u = u[:, ::stride]
        return (u.abs().amax(dim=0) == 0).float().mean()

    def gradient(self, g: torch.Tensor) -> torch.Tensor:
        f, stride = self.sample(g)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        z = torch.where(finite, f, torch.zeros((), dtype=f.dtype, device=f.device)).abs()
        norm = (z * z).sum().sqrt() * math.sqrt(stride)
        nan, inf = self._nan_inf(g, f, stride)
        return torch.stack([norm, z.sum() / n.clamp_min(1.0), z.amax(), nan, inf])

    # ------------------------------------------------- batched: every layer at once
    @staticmethod
    def _batch_nan_inf(st: "_Staging") -> Tuple[torch.Tensor, torch.Tensor]:
        nan = (torch.isnan(st.buf) & st.valid).sum(1) * st.ratio
        inf = (torch.isinf(st.buf) & st.valid).sum(1) * st.ratio
        # a NaN/Inf outside the sample still shows in the full-tensor sum taken by the hook
        nan = torch.where(st.sampled & torch.isnan(st.full) & (nan == 0), 1.0, nan)
        inf = torch.where(st.sampled & torch.isinf(st.full) & (inf == 0), 1.0, inf)
        return nan, inf

    def batch_activation(self, st: "_Staging", kinds: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
        """``ACT_FIELDS`` of every layer from the staged samples, ``[layers, len(ACT_FIELDS)]``.

        Same definitions as :meth:`activation`, in a few dozen kernels for the whole model
        instead of ~25 per layer (kernel launches, not arithmetic, dominate hook cost on a GPU)."""
        buf = st.buf
        finite = torch.isfinite(buf) & st.valid
        nf = finite.sum(1).to(buf.dtype).clamp_min(1.0)
        z = torch.where(finite, buf, 0.0)
        s2 = (z * z).sum(1)
        mean = z.sum(1) / nf
        std = (s2 / nf - mean * mean).clamp_min(0).sqrt()
        mn = torch.where(finite, buf, float("inf")).amin(1)
        mx = torch.where(finite, buf, float("-inf")).amax(1)
        az = z.abs()
        abs_mean = az.sum(1) / nf
        zero_frac = ((buf == 0) & finite).sum(1) / nf
        sat_tanh = (az >= 0.99).sum(1) / nf
        sat_sigmoid = (((z <= 0.01) | (z >= 0.99)) & finite).sum(1) / nf
        sat = torch.where(kinds == 1, sat_tanh, torch.where(kinds == 2, sat_sigmoid, torch.zeros_like(sat_tanh)))
        nan, inf = self._batch_nan_inf(st)
        norm = s2.sqrt() * st.ratio.sqrt()
        return torch.stack([mean, std, mn, mx, abs_mean, zero_frac, sat, norm, nan, inf, dead], 1)

    def batch_gradient(self, st: "_Staging") -> torch.Tensor:
        """``GRAD_FIELDS`` of every layer from the staged gradient samples, ``[layers, len(GRAD_FIELDS)]``."""
        finite = torch.isfinite(st.buf) & st.valid
        n = finite.sum(1).clamp_min(1)
        z = torch.where(finite, st.buf, 0.0).abs()
        norm = (z * z).sum(1).sqrt() * st.ratio.sqrt()
        nan, inf = self._batch_nan_inf(st)
        return torch.stack([norm, z.sum(1) / n, z.amax(1), nan, inf], 1)

    def histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised histogram over the tensor's own [min, max] (edges = the min/max stats)."""
        f, _ = self.sample(x)
        f = self._f32(f)
        f = torch.where(torch.isfinite(f), f, torch.zeros((), dtype=f.dtype, device=f.device))
        h = torch.histc(f, bins=self.histogram_bins, min=0, max=0)
        return h / h.sum().clamp_min(1.0)


class _Staging:
    """One step's samples of every layer, for batched statistics.

    A hook copies a strided sample of its tensor (at most ``width`` elements; the
    batch dimension is kept so dead units can be measured) into its own row of a
    buffer that is allocated once -- one kernel -- and, when the tensor is
    sampled, one reduction over the whole tensor so a NaN/Inf outside the sample
    is still caught. :meth:`prepare` builds the row masks (cached per shape)."""

    def __init__(self, n_rows: int, width: int, exact_nonfinite: bool = True) -> None:
        self.n, self.width, self.exact = n_rows, width, exact_nonfinite
        self.buf: Optional[torch.Tensor] = None  # [rows, width] float32
        self.full: Optional[torch.Tensor] = None  # [rows] float32 full-tensor sums
        self.shape: List[Tuple[int, int]] = [(0, 0)] * n_rows  # (rows, cols) of each sample; 0 = not observed
        self.numel: List[int] = [0] * n_rows
        self.valid: Optional[torch.Tensor] = None
        self.ratio: Optional[torch.Tensor] = None
        self.sampled: Optional[torch.Tensor] = None
        self._cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def clear(self) -> None:
        self.shape = [(0, 0)] * self.n
        self.numel = [0] * self.n

    def release(self) -> None:
        self.buf = self.full = self.valid = self.ratio = self.sampled = None
        self._cache.clear()
        self.clear()

    def write(self, i: int, x: torch.Tensor) -> None:
        x = x.detach()
        if self.buf is None:
            self.buf = torch.empty(self.n, self.width, dtype=torch.float32, device=x.device)
            self.full = torch.zeros(self.n, dtype=torch.float32, device=x.device)
        N, W = x.numel(), self.width
        if x.dim() >= 2 and 2 <= x.shape[0] <= W // 4:
            B = x.shape[0]
            x2 = x.reshape(B, -1)  # a view for contiguous tensors
            s = math.ceil(B * x2.shape[1] / (W - B)) if x2.numel() > W else 1
            v = x2[:, ::s] if s > 1 else x2
        else:
            flat = x.reshape(-1)
            s = max(1, math.ceil(N / W))
            v = (flat[::s] if s > 1 else flat).unsqueeze(0)
        rows, cols = v.shape
        self.buf[i, :rows * cols].view(rows, cols).copy_(v)
        self.shape[i], self.numel[i] = (rows, cols), N
        if self.exact and rows * cols < N:
            if x.device == self.full.device:
                torch.sum(x, dim=tuple(range(x.dim())), dtype=torch.float32, out=self.full[i])
            else:
                self.full[i] = x.sum(dtype=torch.float32).to(self.full.device)

    def prepare(self) -> None:
        key = tuple(self.shape) + tuple(self.numel)
        hit = self._cache.get(key)
        if hit is None:
            dev = self.buf.device
            n = [r * c for r, c in self.shape]
            lengths = torch.tensor(n, device=dev)
            valid = torch.arange(self.width, device=dev)[None, :] < lengths[:, None]
            ratio = torch.tensor([N / k if k else 1.0 for N, k in zip(self.numel, n)], dtype=torch.float32, device=dev)
            sampled = torch.tensor([self.exact and 0 < k < N for N, k in zip(self.numel, n)], device=dev)
            if len(self._cache) >= 16:
                self._cache.clear()
            hit = self._cache[key] = (valid, ratio, sampled)
        self.valid, self.ratio, self.sampled = hit

    def row(self, i: int) -> torch.Tensor:
        r, c = self.shape[i]
        return self.buf[i, :r * c].view(r, c)


# ================================================================ streaming
class SnapshotStream:
    """Bounded, thread-safe ring buffer of TrainingState snapshots.

    The producer (training) never blocks: when the consumer falls behind, the
    oldest snapshots are dropped and counted in ``dropped``."""

    def __init__(self, capacity: int = 1024) -> None:
        self._q: Deque[TrainingState] = deque(maxlen=capacity)
        self._cv = threading.Condition()
        self._subs: List[Callable[[TrainingState], None]] = []
        self._closed = False
        self.latest: Optional[TrainingState] = None
        self.published = 0
        self.dropped = 0
        self.subscriber_errors = 0

    def publish(self, state: TrainingState) -> None:
        with self._cv:
            if self._q.maxlen is not None and len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(state)
            self.latest = state
            self.published += 1
            subs = list(self._subs)
            self._cv.notify_all()
        for cb in subs:
            try:
                cb(state)
            except Exception:  # a broken consumer must never break training
                self.subscriber_errors += 1

    def subscribe(self, callback: Callable[[TrainingState], None]) -> Callable[[], None]:
        """Call ``callback(state)`` for every new snapshot; returns an unsubscribe function."""
        with self._cv:
            self._subs.append(callback)

        def unsubscribe() -> None:
            with self._cv:
                if callback in self._subs:
                    self._subs.remove(callback)

        return unsubscribe

    def get(self, timeout: Optional[float] = None) -> Optional[TrainingState]:
        """Next unread snapshot (blocks up to ``timeout``); None on timeout or once closed and empty."""
        with self._cv:
            self._cv.wait_for(lambda: bool(self._q) or self._closed, timeout)
            return self._q.popleft() if self._q else None

    def drain(self) -> List[TrainingState]:
        with self._cv:
            out = list(self._q)
            self._q.clear()
            return out

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> Iterator[TrainingState]:
        while True:
            s = self.get()
            if s is None:
                return
            yield s

    def __len__(self) -> int:
        return len(self._q)


# =============================================================== collectors
_KIND_CODES = {"": 0, "tanh": 1, "sigmoid": 2, "relu": 3}


class ActivationCollector:
    """Forward hooks (activations) and output-tensor hooks (gradients) on the selected layers.

    A hook only stages a sample of its tensor (:class:`_Staging`); the statistics of
    all layers are computed together at the end of the step."""

    def __init__(self, layers: List[Tuple[str, nn.Module]], extractor: TensorStatsExtractor, *,
                 track_grads: bool = True, capture_eval: bool = True, exact_nonfinite: bool = True) -> None:
        self.layers = layers
        self.labels = [module_label(n, m) for n, m in layers]
        self.saturation = [_saturation_kind(m) for _, m in layers]
        self.extractor = extractor
        self.track_grads = track_grads
        self.capture_eval = capture_eval
        self.active = True
        self.errors: List[str] = []
        self.handles: List[Any] = []
        L, W = len(layers), extractor.max_elems
        self.stages = {k: _Staging(L, W, exact_nonfinite) for k in ("train", "eval", "grad")}
        self._kinds: Dict[str, torch.Tensor] = {}
        self._nan: Dict[str, torch.Tensor] = {}
        self.reset("train")
        self.reset("eval")

    def reset(self, phase: str) -> None:
        tables: Dict[str, Any] = {"act": set(), "grad": set(), "calls": {}, "numel": {}}
        if phase == "train":
            self.train = tables
            self.stages["train"].clear()
            self.stages["grad"].clear()
        else:
            self.eval = tables
            self.stages["eval"].clear()

    def release(self) -> None:
        """Free the staging buffers (re-allocated lazily)."""
        for st in self.stages.values():
            st.release()
        self._kinds.clear()
        self._nan.clear()

    def attach(self) -> None:
        for i, (_name, module) in enumerate(self.layers):
            self.handles.append(module.register_forward_hook(partial(self._on_forward, i)))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _on_forward(self, i: int, module: nn.Module, inputs: Any, output: Any) -> None:
        try:
            self._forward(i, output)
        except Exception as exc:  # monitoring must never stop the user's training
            self.active = False
            self.errors.append(f"forward hook on {self.labels[i]}: {type(exc).__name__}: {exc}")

    def _forward(self, i: int, output: Any) -> None:
        if _compiling():
            return
        training_pass = torch.is_grad_enabled()
        if training_pass and not self.active:
            return
        if not training_pass and not self.capture_eval:
            return
        t = _first_float_tensor(output)
        if t is None:
            return
        tab = self.train if training_pass else self.eval
        with torch.no_grad():
            self.stages["train" if training_pass else "eval"].write(i, t)
        tab["act"].add(i)
        tab["calls"][i] = tab["calls"].get(i, 0) + 1
        tab["numel"][i] = t.numel()
        if training_pass and self.track_grads and t.requires_grad:
            t.register_hook(partial(self._on_grad, i))

    def _on_grad(self, i: int, g: torch.Tensor) -> None:
        try:
            if self.active and g is not None:
                with torch.no_grad():
                    self.stages["grad"].write(i, g)
                self.train["grad"].add(i)
        except Exception as exc:
            self.active = False
            self.errors.append(f"gradient hook on {self.labels[i]}: {type(exc).__name__}: {exc}")
        return None  # never modifies the gradient

    # ------------------------------------------------- end of step: all layers at once
    def _kind_codes(self, dev: torch.device) -> torch.Tensor:
        key = str(dev)
        if key not in self._kinds:
            self._kinds[key] = torch.tensor([_KIND_CODES[k] for k in self.saturation], device=dev)
        return self._kinds[key]

    def _dead_units(self, st: _Staging, observed: set) -> torch.Tensor:
        """dead_unit_frac per layer (NaN for non-ReLU layers): units zero for every sample of the batch."""
        dev = st.buf.device
        key = str(dev)
        if key not in self._nan:
            self._nan[key] = torch.full((), float("nan"), device=dev)
        nan = self._nan[key]
        vals = []
        for i, kind in enumerate(self.saturation):
            if kind == "relu" and i in observed and st.shape[i][0] >= MIN_DEAD_UNIT_BATCH:
                vals.append((st.row(i).abs().amax(0) == 0).float().mean())
            else:
                vals.append(nan)
        return torch.stack(vals)

    def activation_stats(self, phase: str) -> Optional[torch.Tensor]:
        tab = self.train if phase == "train" else self.eval
        st = self.stages[phase]
        if not tab["act"] or st.buf is None:
            return None
        st.prepare()
        return self.extractor.batch_activation(st, self._kind_codes(st.buf.device), self._dead_units(st, tab["act"]))

    def gradient_stats(self) -> Optional[torch.Tensor]:
        st = self.stages["grad"]
        if not self.train["grad"] or st.buf is None:
            return None
        st.prepare()
        return self.extractor.batch_gradient(st)

    def histograms(self, phase: str) -> Dict[int, torch.Tensor]:
        tab = self.train if phase == "train" else self.eval
        st, bins, out = self.stages[phase], self.extractor.histogram_bins, {}
        for i in sorted(tab["act"]):
            row = st.row(i).reshape(-1)
            f = torch.where(torch.isfinite(row), row, 0.0)
            h = torch.histc(f, bins=bins, min=0, max=0)
            out[i] = h / h.sum().clamp_min(1.0)
        return out


class ParameterCollector:
    """Optimizer-step hooks: per-parameter gradient norms, weight norms and weight deltas.

    Tracks every named parameter of the model (not only those given to the
    optimizer), so a trainable layer left out of the optimizer shows a zero delta.
    Uses fused multi-tensor (``torch._foreach_*``) kernels, so the cost is a handful
    of launches per step whatever the number of parameters."""

    def __init__(self, named_params: List[Tuple[str, nn.Parameter]], max_elems: int = 1 << 20) -> None:
        self.names = [n for n, _ in named_params]
        self.params = [p for _, p in named_params]
        self.max_elems = max_elems
        self._shadow: List[Optional[torch.Tensor]] = [None] * len(self.params)
        self._strides = [1] * len(self.params)
        self._scale: Optional[torch.Tensor] = None  # sqrt(stride) per parameter
        self._pre: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.last_has_grad: List[bool] = [False] * len(self.params)
        self._captured = False
        self.device: Optional[torch.device] = self.params[0].device if self.params else None
        self._fused = len({p.device for p in self.params}) <= 1  # foreach kernels need one device

    def _sample(self, i: int, t: torch.Tensor) -> torch.Tensor:
        flat = t.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        s = self._strides[i]
        return flat[::s] if s > 1 else flat

    def _norms(self, ts: List[torch.Tensor], ord: float = 2.0) -> torch.Tensor:
        """L2 norm (``ord=2``) or max |x| (``ord=inf``) of each tensor, as one float32 vector.

        CUDA uses the fused ``torch._foreach_norm``. On CPU and MPS per-tensor dot
        products / multiply-and-sum are much faster (MPS's norm kernel is ~20x slower
        than a multiply-and-sum; measured in docs/overhead_benchmark.md)."""
        if not ts:
            return torch.empty(0, device=self.device)
        kind = self.device.type if self.device is not None else "cpu"
        if self._fused and kind == "cuda":
            out = torch._foreach_norm(ts, ord)
            return torch.stack([t if t.dtype == torch.float32 else t.float() for t in out])
        vals = []
        for t in ts:
            f = t.reshape(-1)
            if f.dtype not in (torch.float32, torch.float64):
                f = f.float()
            if ord == 2.0:
                v = torch.dot(f, f) if kind == "cpu" else (f * f).sum()
            else:
                v = f.abs().amax() if f.numel() else torch.zeros((), device=f.device)
            vals.append(v if self._fused else v.to(self.device))
        out = torch.stack([v if v.dtype == torch.float32 else v.float() for v in vals])
        return out.sqrt() if ord == 2.0 else out

    @staticmethod
    def _dense(g: torch.Tensor) -> torch.Tensor:
        """The stored values of a sparse gradient (its norm and max are the same), else the gradient."""
        g = g.detach()
        return g.coalesce().values() if g.is_sparse else g

    @torch.no_grad()
    def refresh_grads(self) -> None:
        """Re-measure gradients after step(closure): the closure computed them after the pre-hook ran."""
        if self._pre is None:
            return
        self.last_has_grad = [p.grad is not None for p in self.params]
        grads = [self._dense(p.grad) for p in self.params if p.grad is not None]
        self._pre = (self._norms(grads), self._norms(grads, float("inf")), self._pre[2])

    @torch.no_grad()
    def norms(self) -> None:
        """Per-parameter gradient L2 norm and max |gradient| (exact), and weight norm."""
        self.last_has_grad = [p.grad is not None for p in self.params]
        grads = [self._dense(p.grad) for p in self.params if p.grad is not None]
        for i, p in enumerate(self.params):
            self._ensure_shadow(i, p)
        self._pre = (self._norms(grads), self._norms(grads, float("inf")),
                     self._norms([p.detach() for p in self.params]))

    def _ensure_shadow(self, i: int, p: torch.Tensor) -> None:
        if self._shadow[i] is None:
            self._strides[i] = max(1, math.ceil(p.numel() / self.max_elems))
            self._shadow[i] = torch.empty_like(self._sample(i, p))
            self._scale = None

    @torch.no_grad()
    def capture(self) -> None:
        """Remember the current weights (a strided sample, into buffers allocated once)."""
        for i, p in enumerate(self.params):
            self._ensure_shadow(i, p)
        samples = [self._sample(i, p) for i, p in enumerate(self.params)]
        if samples and self._fused and hasattr(torch, "_foreach_copy_"):
            torch._foreach_copy_(self._shadow, samples)  # the only weight copy
        else:
            for buf, smp in zip(self._shadow, samples):
                buf.copy_(smp)
        self._captured = True

    def before_step(self) -> None:
        self.norms()
        self.capture()

    def release(self) -> None:
        """Free the weight-sample buffers (re-allocated lazily if monitoring resumes)."""
        self._shadow = [None] * len(self.params)
        self._scale = None
        self._pre = None
        self._captured = False

    @torch.no_grad()
    def after_step(self) -> torch.Tensor:
        """``[grad_norm..., grad_abs_max...]`` for the parameters with a gradient, then ``[weight_norm..., delta_norm...]``
        for every parameter; the delta is measured against the last capture()."""
        if self._pre is None:
            self.norms()
        gnorm, gmax, wnorm = self._pre  # type: ignore[misc]
        P = len(self.params)
        if self._captured and P:
            samples = [self._sample(i, p) for i, p in enumerate(self.params)]
            if self._fused:
                torch._foreach_sub_(self._shadow, samples)  # in place: old - new
            else:
                for buf, smp in zip(self._shadow, samples):
                    buf.sub_(smp)
            if self._scale is None:
                self._scale = torch.tensor([math.sqrt(s) for s in self._strides], dtype=torch.float32,
                                           device=self.device)
            dnorm = self._norms(self._shadow) * self._scale  # type: ignore[arg-type]
        else:
            dnorm = torch.full((P,), float("nan"), device=self.device)
        self._pre = None
        self._captured = False
        return torch.cat([gnorm, gmax, wnorm, dnorm])


class _BackwardPatch:
    """One shared ``torch.Tensor.backward`` wrapper for every active :class:`LossCollector`.

    Collectors register and unregister in any order (several models, e.g. a GAN,
    or nested sessions); the original method is restored when the last one leaves,
    and only if nobody else has patched ``backward`` on top of us since."""

    lock = threading.Lock()
    collectors: List["LossCollector"] = []
    orig: Any = None
    wrapper: Any = None
    local = threading.local()

    scaler_orig: Any = None
    scaler_wrapper: Any = None

    @classmethod
    def add(cls, c: "LossCollector") -> None:
        with cls.lock:
            if c not in cls.collectors:
                cls.collectors.append(c)
            if cls.wrapper is None or torch.Tensor.backward is not cls.wrapper:
                cls.orig = torch.Tensor.backward
                cls.wrapper = cls._make(cls.orig)
                torch.Tensor.backward = cls.wrapper  # type: ignore[assignment]
            scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
            if scaler is not None and cls.scaler_wrapper is None:
                cls.scaler_orig = scaler.scale
                cls.scaler_wrapper = cls._make_scale(cls.scaler_orig)
                scaler.scale = cls.scaler_wrapper

    @staticmethod
    def _make_scale(orig: Any) -> Any:
        def scale(self, outputs):  # type: ignore[no-untyped-def]
            out = orig(self, outputs)
            factor = getattr(self, "_scale", None)
            if isinstance(out, torch.Tensor) and out is not outputs and isinstance(factor, torch.Tensor):
                out._sba_loss_scale = factor  # backward() divides it out again
            return out

        return scale

    @classmethod
    def remove(cls, c: "LossCollector") -> None:
        with cls.lock:
            if c in cls.collectors:
                cls.collectors.remove(c)
            if not cls.collectors and cls.wrapper is not None and torch.Tensor.backward is cls.wrapper:
                torch.Tensor.backward = cls.orig  # type: ignore[assignment]
                cls.orig = cls.wrapper = None
            scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
            if not cls.collectors and scaler is not None and cls.scaler_wrapper is not None \
                    and scaler.scale is cls.scaler_wrapper:
                scaler.scale = cls.scaler_orig
                cls.scaler_orig = cls.scaler_wrapper = None

    @classmethod
    def _make(cls, orig: Any) -> Any:
        def backward(t, *a, **kw):  # type: ignore[no-untyped-def]
            nested = getattr(cls.local, "inside", False)  # backward() called from inside a backward pass
            if not nested and cls.collectors and not _compiling() and t.numel() == 1 and t.is_floating_point():
                v = t.detach().reshape(()).float()
                factor = getattr(t, "_sba_loss_scale", None)
                if factor is not None:
                    v = v / factor.detach().reshape(()).float().to(v.device)
                for c in list(cls.collectors):
                    c.values.append(v)
            cls.local.inside = True
            try:
                return orig(t, *a, **kw)
            finally:
                cls.local.inside = nested

        return backward


class LossCollector:
    """Captures every scalar passed to ``backward()`` as a device tensor (no sync)."""

    def __init__(self) -> None:
        self.values: List[torch.Tensor] = []
        self._installed = False

    def install(self) -> None:
        if not self._installed:
            _BackwardPatch.add(self)
            self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            _BackwardPatch.remove(self)
            self._installed = False

    def observe(self, loss: Union[torch.Tensor, float]) -> None:
        if isinstance(loss, torch.Tensor):
            self.values.append(loss.detach().reshape(()).float())
        else:
            self.values.append(torch.tensor(float(loss)))


# ============================================================= orchestrator
@dataclass
class HookConfig:
    every_n_steps: int = 1
    layers: Union[str, Callable[[str, nn.Module], bool]] = "auto"  # "auto" or a predicate(name, module)
    max_layers: int = 256
    max_elems: int = 1 << 14  # per activation/gradient sample; larger tensors are sampled (strided view)
    max_param_elems: int = 1 << 20  # per weight tensor, for the weight-delta sample
    histogram_bins: int = 0  # > 0 adds normalised histograms to extra["histograms"]
    track_activation_grads: bool = True
    capture_backward_loss: bool = True
    eval_snapshots: str = "first"  # "first" per validation phase | "every" no-grad forward | "off"
    # "exact": one extra reduction over every sampled tensor, so a NaN/Inf anywhere is counted;
    # "sampled": only inside the sample (cheaper on GPUs; NaN usually spreads to the sample within a layer or two)
    nonfinite_check: str = "exact"
    max_loss_history: Optional[int] = 10_000  # loss values kept in loss_history (None = all)
    stream_capacity: int = 1024
    max_pending: int = 4  # device-to-host copies allowed in flight before waiting
    loss_z: float = 6.0
    loss_warmup: int = 10


def select_layers(model: nn.Module, rule: Union[str, Callable[[str, nn.Module], bool]] = "auto",
                  max_layers: int = 256) -> List[Tuple[str, nn.Module]]:
    """Leaf modules (plus attention / RNN modules as single units), in registration order."""
    out: List[Tuple[str, nn.Module]] = []
    units: List[str] = []
    for name, m in model.named_modules():
        name = name.replace("_orig_mod.", "").replace("_orig_mod", "")
        if not name:
            continue
        if callable(rule):
            keep = bool(rule(name, m))
        else:
            if isinstance(m, _SKIP_TYPES) or any(name.startswith(u + ".") for u in units):
                continue
            unit = isinstance(m, _UNIT_TYPES)
            keep = unit or next(m.children(), None) is None
            if unit:
                units.append(name)
        if keep:
            out.append((name, m))
        if len(out) >= max_layers:
            break
    return out


class RuntimeHooks:
    """Attach non-intrusive hooks to ``model`` (and ``optimizer``) and stream TrainingState snapshots.

    Usage::

        hooks = RuntimeHooks(model, optimizer, invariants=get_default_registry()).attach()
        for x, y in loader:                     # the training loop is unchanged
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()                    # a TrainingState is produced here
        hooks.detach()
        print(hooks.failures)                   # [(step, InvariantResult), ...]
        states = hooks.stream.drain()           # or hooks.stream.get() from an agent thread

    Without an optimizer (manual weight updates) call ``hooks.end_step()`` after each step.
    Validation passes (forward under ``torch.no_grad()``) produce ``is_eval_phase=True`` snapshots."""

    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                 config: Optional[HookConfig] = None, *, invariants: Optional[List[Invariant]] = None,
                 stream: Optional[SnapshotStream] = None,
                 on_failure: Optional[Callable[[TrainingState, InvariantResult], None]] = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or HookConfig()
        self.stream = stream or SnapshotStream(self.config.stream_capacity)
        self.invariants = invariants
        self.on_failure = on_failure
        self.failures: List[Tuple[int, InvariantResult]] = []
        self.extractor = TensorStatsExtractor(self.config.max_elems, self.config.histogram_bins)
        self.layers = select_layers(model, self.config.layers, self.config.max_layers)
        self.acts = ActivationCollector(self.layers, self.extractor, track_grads=self.config.track_activation_grads,
                                        capture_eval=self.config.eval_snapshots != "off",
                                        exact_nonfinite=self.config.nonfinite_check != "sampled")
        named = [(n.replace("_orig_mod.", ""), p) for n, p in model.named_parameters()]
        seen = {id(p) for _, p in named}
        if optimizer is not None:
            extra = [p for g in optimizer.param_groups for p in g["params"] if id(p) not in seen]
            named += [(f"optimizer_param{i}", p) for i, p in enumerate(extra)]
        self.param_coll = ParameterCollector(named, self.config.max_param_elems)
        self.losses = LossCollector()
        self.loss_history: List[float] = []
        self._loss_det = EmaZScore(alpha=0.05, warmup=self.config.loss_warmup, z=self.config.loss_z)
        self._handles: List[Any] = []
        self._steps_done = 0
        self._paused = False
        self._broken = False
        self._closure = False
        self._attached = False
        self._eval_taken = False
        self._pending: Deque[Tuple[Any, torch.Tensor, Dict[str, Any]]] = deque()
        self._free: List[torch.Tensor] = []
        self._nan_rows: Dict[Tuple[int, str], torch.Tensor] = {}

    # ---------------------------------------------------------------- lifecycle
    @property
    def step(self) -> int:
        """Number of optimizer steps completed so far."""
        return self._steps_done

    def _monitoring(self) -> bool:
        return not self._paused and self._steps_done % max(1, self.config.every_n_steps) == 0

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Stop monitoring from the next step on (hooks stay installed and cost almost nothing)."""
        self._paused = True
        self.acts.active = False

    @property
    def errors(self) -> List[str]:
        """Internal errors that switched monitoring off (training was never interrupted)."""
        return self.acts.errors

    def resume(self) -> None:
        if self._broken:
            return
        self._paused = False
        self.acts.active = self._monitoring()

    def set_every_n_steps(self, n: int) -> None:
        """Change the monitoring rate while attached (used by the overhead governor)."""
        self.config.every_n_steps = max(1, int(n))
        self.acts.active = self._monitoring()

    def attach(self) -> "RuntimeHooks":
        if self._attached:
            return self
        self.acts.attach()
        self.acts.active = self._monitoring()
        self._handles.append(self.model.register_forward_hook(self._on_root_forward))
        if self.config.capture_backward_loss:
            self.losses.install()
        if self.optimizer is not None:
            self._handles.append(self.optimizer.register_step_pre_hook(self._on_opt_pre))
            self._handles.append(self.optimizer.register_step_post_hook(self._on_opt_post))
        elif self._monitoring():
            self.param_coll.capture()  # manual updates: deltas are measured from here
        self._attached = True
        return self

    def detach(self) -> None:
        """Remove every hook and free every device buffer the hooks allocated (no memory is held after this)."""
        if not self._attached:
            return
        self.flush()
        self.acts.detach()
        self.losses.uninstall()
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False
        self.param_coll.release()
        self.acts.reset("train")
        self.acts.reset("eval")
        self.acts.release()
        self.losses.values.clear()
        self._free.clear()
        self._nan_rows.clear()

    def __enter__(self) -> "RuntimeHooks":
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    def observe_loss(self, loss: Union[torch.Tensor, float]) -> None:
        """Report a loss explicitly (if backward() is not used or capture_backward_loss is off)."""
        self.losses.observe(loss)

    # -------------------------------------------------------------- training
    def _fail(self, where: str, exc: BaseException) -> None:
        """Turn monitoring off for good after an internal error; training carries on untouched."""
        self.errors.append(f"{where}: {type(exc).__name__}: {exc}")
        self._broken = True
        self.pause()

    def _on_opt_pre(self, opt: Any, args: Any, kwargs: Any) -> None:
        # PyTorch passes (optimizer, *step_args), so a closure is any callable among them
        self._closure = any(callable(a) for a in (args or ())) or callable((kwargs or {}).get("closure"))
        if self._monitoring():
            try:
                self.param_coll.before_step()
            except Exception as exc:
                self._fail("optimizer pre-hook", exc)

    def _on_opt_post(self, opt: Any, args: Any, kwargs: Any) -> None:
        rows = None
        if self._monitoring():
            try:
                if self._closure:
                    self.param_coll.refresh_grads()
                rows = self.param_coll.after_step()
            except Exception as exc:
                self._fail("optimizer post-hook", exc)
        try:
            self._complete_step(rows)
        except Exception as exc:
            self._fail("snapshot", exc)

    def end_step(self) -> None:
        """Mark the end of a training step when no optimizer hook is available (manual updates).

        Weight deltas are measured against the weights at the end of the previous step."""
        rows = None
        if self._monitoring():
            self.param_coll.norms()
            rows = self.param_coll.after_step()
        self._complete_step(rows)
        if self.optimizer is None and self._monitoring():
            self.param_coll.capture()

    def _complete_step(self, param_rows: Optional[torch.Tensor]) -> None:
        self._steps_done += 1
        if param_rows is not None:
            self._finalize("train", param_rows)
        self.acts.reset("train")
        self.losses.values.clear()
        self.acts.active = self._monitoring()
        self._eval_taken = False
        self._poll(block=False)

    # ---------------------------------------------------------------- eval
    def _on_root_forward(self, module: nn.Module, inputs: Any, output: Any) -> None:
        try:
            self._root_forward()
        except Exception as exc:
            self._fail("validation snapshot", exc)

    def _root_forward(self) -> None:
        if _compiling() or torch.is_grad_enabled() or self.config.eval_snapshots == "off":
            return
        if self.config.eval_snapshots == "first" and self._eval_taken:
            self.acts.reset("eval")
            return
        self._eval_taken = True
        self._finalize("eval", None)
        self.acts.reset("eval")

    # -------------------------------------------------------------- packing
    def _device(self) -> torch.device:
        if self.param_coll.device is not None:
            return self.param_coll.device
        for st in self.acts.stages.values():
            if st.buf is not None:
                return st.buf.device
        return torch.device("cpu")

    def _nan_row(self, n: int, dev: torch.device) -> torch.Tensor:
        key = (n, str(dev))
        if key not in self._nan_rows:
            self._nan_rows[key] = torch.full((n,), float("nan"), device=dev)
        return self._nan_rows[key]

    def _block(self, stats: Optional[torch.Tensor], width: int, dev: torch.device) -> torch.Tensor:
        L = len(self.layers)
        if L == 0 or width == 0:
            return torch.empty(0, device=dev)
        if stats is None:
            return self._nan_row(L * width, dev)
        stats = stats.reshape(-1)
        return stats if stats.device == dev else stats.to(dev, non_blocking=True)

    def _finalize(self, phase: str, param_vec: Optional[torch.Tensor]) -> None:
        dev = self._device()
        tab = self.acts.train if phase == "train" else self.acts.eval
        bins = self.extractor.histogram_bins
        hist_rows: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            losses = [v if v.device == dev else v.to(dev, non_blocking=True) for v in self.losses.values]
            loss = torch.stack(losses).mean().reshape(1) if (losses and phase == "train") else self._nan_row(1, dev)
            parts = [
                loss,
                self._block(self.acts.activation_stats(phase), len(ACT_FIELDS), dev),
                self._block(self.acts.gradient_stats() if phase == "train" else None, len(GRAD_FIELDS), dev),
            ]
            if bins:
                hist_rows = self.acts.histograms(phase)
                parts.append(torch.cat([hist_rows[i].to(dev) if i in hist_rows else self._nan_row(bins, dev)
                                        for i in range(len(self.layers))]) if self.layers else torch.empty(0, device=dev))
            if param_vec is not None:
                parts.append(param_vec.to(dev).float())
            packed = torch.cat(parts)
        host = self._host_buffer(packed)
        host.copy_(packed, non_blocking=packed.is_cuda)
        event = None
        if packed.is_cuda:
            event = torch.cuda.Event()
            event.record()
        meta = {
            "phase": phase,
            "step": self._steps_done,
            "wall_time": time.time(),
            "n_backward": len(self.losses.values) if phase == "train" else 0,
            "calls": dict(tab["calls"]),
            "numel": dict(tab["numel"]),
            "has_act": set(tab["act"]),
            "has_grad": set(tab["grad"]),
            "has_hist": set(hist_rows),
            "has_params": param_vec is not None,
            "has_param_grad": list(self.param_coll.last_has_grad),
            "lrs": [float(g.get("lr", float("nan"))) for g in self.optimizer.param_groups] if self.optimizer else [],
            "module_modes": {module_label(n.replace("_orig_mod.", "").replace("_orig_mod", ""), m):
                             ("train" if m.training else "eval") for n, m in self.model.named_modules()},
            "requires_grad": {n: bool(p.requires_grad) for n, p in zip(self.param_coll.names, self.param_coll.params)},
        }
        self._pending.append((event, host, meta))
        while len(self._pending) > self.config.max_pending:
            self._publish(*self._pending.popleft())

    def _host_buffer(self, packed: torch.Tensor) -> torch.Tensor:
        n = packed.numel()
        for i, b in enumerate(self._free):
            if b.numel() >= n:
                return self._free.pop(i)[:n]
        pin = packed.is_cuda and torch.cuda.is_available()
        return torch.empty(max(n, 64), dtype=torch.float32, pin_memory=pin)[:n]

    def _poll(self, block: bool) -> None:
        while self._pending:
            event, host, meta = self._pending[0]
            if not block and event is not None and not event.query():
                return
            self._pending.popleft()
            self._publish(event, host, meta)

    def flush(self) -> None:
        """Publish every pending snapshot (waits for in-flight device-to-host copies)."""
        self._poll(block=True)

    @property
    def latest(self) -> Optional[TrainingState]:
        self._poll(block=False)
        return self.stream.latest

    def _publish(self, event: Any, host: torch.Tensor, meta: Dict[str, Any]) -> None:
        if event is not None:
            event.synchronize()
        vals = host.tolist()
        self._free.append(host)
        state = self._materialize(vals, meta)
        self.stream.publish(state)
        if self.invariants:
            for r in run_all(self.invariants, state):
                if not r.passed:
                    self.failures.append((state.step, r))
                    if self.on_failure is not None:
                        self.on_failure(state, r)

    # ------------------------------------------------------- materialisation
    def _materialize(self, vals: List[float], meta: Dict[str, Any]) -> TrainingState:
        L, bins = len(self.layers), self.extractor.histogram_bins
        pos = 0
        loss = vals[pos]
        pos += 1
        act = vals[pos: pos + L * len(ACT_FIELDS)]
        pos += L * len(ACT_FIELDS)
        grd = vals[pos: pos + L * len(GRAD_FIELDS)]
        pos += L * len(GRAD_FIELDS)
        hist = vals[pos: pos + (L * bins if bins else 0)]
        pos += L * bins if bins else 0
        prm = vals[pos:]

        activation_stats: Dict[str, Dict[str, float]] = {}
        histograms: Dict[str, Dict[str, Any]] = {}
        for i, label in enumerate(self.acts.labels):
            if i not in meta["has_act"] and i not in meta["has_grad"]:
                continue
            st: Dict[str, float] = {"calls": float(meta["calls"].get(i, 0)), "numel": float(meta["numel"].get(i, 0))}
            if i in meta["has_act"]:
                st.update(zip(ACT_FIELDS, act[i * len(ACT_FIELDS): (i + 1) * len(ACT_FIELDS)]))
                if st["dead_unit_frac"] != st["dead_unit_frac"]:  # NaN: not a ReLU layer, not measured
                    del st["dead_unit_frac"]
            if i in meta["has_grad"]:
                st.update(zip(GRAD_FIELDS, grd[i * len(GRAD_FIELDS): (i + 1) * len(GRAD_FIELDS)]))
            activation_stats[label] = st
            if bins and i in meta["has_hist"]:
                histograms[label] = {"min": st.get("min"), "max": st.get("max"), "density": hist[i * bins: (i + 1) * bins]}

        grad_norms: Dict[str, float] = {}
        weight_deltas: Dict[str, float] = {}
        weight_norms: Dict[str, float] = {}
        grad_abs_max: Dict[str, float] = {}
        if meta["has_params"]:
            has, P = meta["has_param_grad"], len(self.param_coll.names)
            ng = sum(has)
            gnv, gmv, wnv, dnv = prm[:ng], prm[ng:2 * ng], prm[2 * ng:2 * ng + P], prm[2 * ng + P:2 * ng + 2 * P]
            k = 0
            for j, name in enumerate(self.param_coll.names):
                if has[j]:
                    grad_norms[name] = gnv[k]  # NaN/Inf kept as-is: the gradient invariant flags them
                    grad_abs_max[name] = gmv[k]
                    k += 1
                weight_deltas[name] = dnv[j]
                weight_norms[name] = wnv[j]

        extra: Dict[str, Any] = {"source": "silent_bug_agent.runtime.hooks", "wall_time": meta["wall_time"],
                                 "lrs": meta["lrs"]}
        if meta["phase"] == "train":
            loss_v: Optional[float] = loss if meta["n_backward"] > 0 else None
            if loss_v is not None:
                self.loss_history.append(loss_v)
                cap = self.config.max_loss_history
                if cap is not None and len(self.loss_history) > cap:
                    del self.loss_history[: len(self.loss_history) - cap]
            z, spike = (self._loss_det.update(loss_v) if loss_v is not None else (None, False))
            extra["loss"] = {
                "value": loss_v,
                "ema": self._loss_det.mean if self._loss_det.n else None,
                "z": z if z is not None and math.isfinite(z) else None,
                "spike": bool(spike),
                "n_backward": meta["n_backward"],
            }
            finite_g = [v for v in grad_norms.values() if math.isfinite(v)]
            extra["global_grad_norm"] = math.sqrt(sum(v * v for v in finite_g)) if finite_g else None
            wsq = sum(v * v for v in weight_norms.values() if math.isfinite(v))
            dsq = sum(v * v for v in weight_deltas.values() if math.isfinite(v))
            extra["global_weight_norm"] = math.sqrt(wsq)
            extra["global_delta_norm"] = math.sqrt(dsq)
            extra["update_ratio"] = math.sqrt(dsq) / math.sqrt(wsq) if wsq > 0 else None
            extra["weight_norms"] = weight_norms
            extra["grad_abs_max"] = grad_abs_max
        if histograms:
            extra["histograms"] = histograms

        return TrainingState(
            step=meta["step"],
            loss_history=list(self.loss_history),
            grad_norms=grad_norms,
            weight_deltas=weight_deltas,
            requires_grad=meta["requires_grad"],
            activation_stats=activation_stats,
            module_modes=meta["module_modes"],
            is_eval_phase=meta["phase"] == "eval",
            extra=extra,
        )
