"""Read-only weight inspection.

Looks at the numbers inside a model for signs of a training bug that code analysis
cannot see: non-finite or exploded values, a weight matrix stuck at zero, biases so
negative that a ReLU can never fire, and matrices that have collapsed to a few
directions. It only reads. It never changes a model, a checkpoint or a script.

Weights come from one of two places:

* a checkpoint file, loaded with `weights_only=True` so a pickle cannot run code;
* a training script, run for a few steps in the validation sandbox (a separate
  process with a time limit), which writes its trained tensors to a temp file.

A freshly initialised model says nothing about a training bug, so the script path
always inspects weights *after* some optimizer steps.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .schemas import WeightAnomalyType, WeightFinding, WeightReport

# Running statistics of normalisation layers follow the data's scale, so a large or zero value is not a bug.
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


class WeightInspector:
    """Measures weight tensors and reports anomalies. It does not modify anything."""

    def __init__(
        self,
        *,
        min_effective_rank_ratio: float = 0.15,
        max_kurtosis: float = 30.0,
        outlier_sigmas: float = 100.0,
        extreme_bias_threshold: float = -15.0,
        max_abs_value: float = 1e4,
        svd_max_dim: int = 1024,
    ) -> None:
        self.min_effective_rank_ratio = min_effective_rank_ratio
        self.max_kurtosis = max_kurtosis
        self.outlier_sigmas = outlier_sigmas
        self.extreme_bias_threshold = extreme_bias_threshold
        self.max_abs_value = max_abs_value
        self.svd_max_dim = svd_max_dim

    # ---------------------------------------------------------------- entry
    def inspect_model(self, model: nn.Module, model_name: str = "Model") -> WeightReport:
        """Inspect a live model. Note that an untrained model only shows its initialisation."""
        return self.inspect_state_dict(model.state_dict(), model_name=model_name, source="live model")

    def inspect_checkpoint(self, path: Union[str, Path]) -> WeightReport:
        """Inspect a saved state_dict (.pt, .pth, .bin, .ckpt or .safetensors) without running any code in it."""
        file_path = Path(path).resolve()
        report = WeightReport(model_name=file_path.stem, source=f"checkpoint {file_path.name}")
        if not file_path.is_file():
            report.diagnostics.append(f"Checkpoint not found: {file_path}")
            return report
        try:
            if file_path.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    report.diagnostics.append("Reading .safetensors needs the safetensors package: pip install safetensors")
                    return report
                loaded: Any = load_file(str(file_path))
            else:
                loaded = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - every unreadable file is reported, not raised
            report.diagnostics.append(
                f"Could not read the checkpoint safely ({type(exc).__name__}: {str(exc)[:200]}). Only plain tensor "
                "checkpoints are loaded, because a pickled model can run code when opened. Save one with "
                "torch.save(model.state_dict(), path).")
            return report
        state_dict = self._tensors_of(loaded)
        if not state_dict:
            report.diagnostics.append("The checkpoint holds no tensors under the usual keys (state_dict, model).")
            return report
        return self.inspect_state_dict(state_dict, model_name=file_path.stem, source=report.source)

    @staticmethod
    def _tensors_of(loaded: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(loaded, dict):
            return {}
        for key in STATE_DICT_KEYS:
            inner = loaded.get(key)
            if isinstance(inner, dict) and any(isinstance(v, torch.Tensor) for v in inner.values()):
                loaded = inner
                break
        return {str(k): v for k, v in loaded.items() if isinstance(v, torch.Tensor)}

    def inspect_state_dict(self, state_dict: Dict[str, torch.Tensor], model_name: str = "Model",
                           source: str = "state_dict") -> WeightReport:
        report = WeightReport(model_name=model_name, source=source)
        skipped_large = 0
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or name.endswith(BUFFER_SUFFIXES):
                continue
            if not tensor.is_floating_point() or tensor.is_sparse:
                continue
            report.inspected = True
            report.total_parameters += tensor.numel()
            values = tensor.detach().to("cpu", torch.float32)
            shape = tuple(tensor.shape)

            bad = int((~torch.isfinite(values)).sum())
            if bad:
                report.findings.append(WeightFinding(
                    name, shape, WeightAnomalyType.NAN_INF, "critical", float(bad), 0.0,
                    f"{bad} of {values.numel()} values are NaN or infinite, so every output that uses this tensor "
                    "is NaN.",
                    {"non_finite": bad},
                    "Training diverged before this point: lower the learning rate, clip gradients, and check the "
                    "loss for log(0) or a division by zero."))
                continue

            largest = float(values.abs().max())
            if largest > self.max_abs_value:
                report.findings.append(WeightFinding(
                    name, shape, WeightAnomalyType.EXPLODED, "warning", largest, self.max_abs_value,
                    f"The largest value is {largest:.3g}. Healthy weights are rarely above {self.max_abs_value:g}; "
                    "this is what an exploding update looks like just before NaN.",
                    {"max_abs": largest},
                    "Lower the learning rate or add gradient clipping, then inspect again."))

            if name.endswith("bias") and values.dim() == 1:
                finding = self._check_bias(name, values)
                if finding:
                    report.findings.append(finding)
            if values.dim() >= 2:
                findings, summary, too_large = self._check_matrix(name, values)
                report.findings.extend(findings)
                skipped_large += too_large
                if summary:
                    report.layer_summaries[name] = summary
        if not report.inspected:
            report.diagnostics.append("No floating-point tensors were found to inspect.")
        if skipped_large:
            report.diagnostics.append(f"{skipped_large} matrices are larger than {self.svd_max_dim} on a side, so "
                                      "their rank was not measured.")
        return report

    # --------------------------------------------------------------- checks
    def _check_bias(self, name: str, bias: torch.Tensor) -> Optional[WeightFinding]:
        dead = int((bias <= self.extreme_bias_threshold).sum())
        if not dead:
            return None
        share = 100.0 * dead / bias.numel()
        severity = "critical" if share >= 50 else "warning" if share >= 5 else "info"
        lowest = float(bias.min())
        return WeightFinding(
            name, tuple(bias.shape), WeightAnomalyType.BIAS_COLLAPSE, severity, lowest, self.extreme_bias_threshold,
            f"{dead} of {bias.numel()} biases ({share:.1f}%) are at or below {self.extreme_bias_threshold:g} "
            f"(lowest {lowest:.2f}). If a ReLU follows this layer, those units output zero for any ordinary input "
            "and get no gradient, so they cannot recover.",
            {"min_bias": lowest, "max_bias": float(bias.max()), "mean_bias": float(bias.mean()), "pct_dead": share},
            "Find where the bias is set or driven this low (an initialisation, or one huge update) and fix that "
            "in the code; `fix` handles the initialisation case.")

    def _check_matrix(self, name: str, tensor: torch.Tensor
                      ) -> Tuple[List[WeightFinding], Optional[Dict[str, Any]], int]:
        """Returns the findings, a summary of the matrix, and 1 if it was too large for the rank check."""
        findings: List[WeightFinding] = []
        shape = tuple(tensor.shape)
        matrix = tensor.reshape(tensor.shape[0], -1)
        rows, cols = matrix.shape
        min_dim = min(rows, cols)
        frob = float(torch.linalg.matrix_norm(matrix))
        if frob == 0.0:
            findings.append(WeightFinding(
                name, shape, WeightAnomalyType.ALL_ZERO, "warning", 0.0, 0.0,
                "Every value is zero, so this layer outputs a constant and passes no signal on.",
                {},
                "Some designs start a layer at zero on purpose. After training steps it should have moved: check "
                "that the layer is in the optimizer and receives gradient."))
            return findings, {"shape": [rows, cols], "frob_norm": 0.0}, 0
        summary: Dict[str, Any] = {"shape": [rows, cols], "frob_norm": round(frob, 4)}

        if matrix.numel() >= 64:
            centered = matrix - matrix.mean()
            variance = float((centered ** 2).mean())
            if variance > 0:
                kurtosis = float((centered ** 4).mean()) / (variance ** 2)
                summary["kurtosis"] = round(kurtosis, 2)
                if kurtosis > self.max_kurtosis:
                    flat = matrix.flatten()
                    median = flat.median()
                    spread = 1.4826 * float((flat - median).abs().median())  # robust estimate of the usual spread
                    extreme = float((flat - median).abs().max())
                    sigmas = extreme / spread if spread > 0 else float("inf")
                    findings.append(WeightFinding(
                        name, shape, WeightAnomalyType.WEIGHT_OUTLIER,
                        "warning" if sigmas >= self.outlier_sigmas else "info", kurtosis, self.max_kurtosis,
                        f"A few values are far outside the rest: the most extreme is {extreme:.3g}, about "
                        f"{sigmas:.0f} times the typical spread (kurtosis {kurtosis:.1f}). Trained weights often "
                        "have heavy tails, so this matters mainly when a single value dominates.",
                        {"kurtosis": kurtosis, "max_deviation": extreme, "typical_spread": spread},
                        "Check for one very large update (no gradient clipping, a learning-rate spike) or a "
                        "hand-written initialisation."))

        if min_dim < 2:
            return findings, summary, 0
        if max(rows, cols) > self.svd_max_dim:
            return findings, summary, 1
        try:
            singular = torch.linalg.svdvals(matrix)
        except Exception:  # noqa: BLE001 - a failed decomposition only loses the rank check
            return findings, summary, 0
        total = float(singular.sum())
        if total <= 0:
            return findings, summary, 0
        share = singular / total
        effective_rank = math.exp(-float((share * torch.log(share + 1e-12)).sum()))
        ratio = effective_rank / min_dim
        smallest = float(singular[-1])
        summary.update({
            "spectral_norm": round(float(singular[0]), 4),
            "condition_number": round(float(singular[0]) / smallest, 2) if smallest > 1e-12 else None,
            "effective_rank": round(effective_rank, 2),
            "rank_ratio": round(ratio, 4),
        })
        if ratio < self.min_effective_rank_ratio and min_dim >= 8:
            findings.append(WeightFinding(
                name, shape, WeightAnomalyType.RANK_COLLAPSE, "critical" if ratio < 0.05 else "warning",
                ratio, self.min_effective_rank_ratio,
                f"This matrix behaves like rank {effective_rank:.1f} out of a possible {min_dim} "
                f"({ratio * 100:.0f}%), so most of its units compute nearly the same thing.",
                dict(summary),
                "Common causes are a learning rate that is too high, dead units in the layer before, or weights "
                "copied across rows. Compare with the same layer at initialisation."))
        return findings, summary, 0


def inspect_script(script: Union[str, Path], *, max_steps: int = 60, timeout: float = 180.0,
                   script_args: Optional[Sequence[str]] = None,
                   inspector: Optional[WeightInspector] = None) -> WeightReport:
    """Train `script` for `max_steps` optimizer steps in the sandbox, then inspect the weights it produced.

    The script runs in a separate process with a time limit, exactly as the validation gates run it, and
    the original file is never written to."""
    from ..validation.sandbox import SandboxConfig, SubprocessSandbox

    path = Path(script).resolve()
    inspector = inspector or WeightInspector()
    report = WeightReport(model_name=path.stem, source=f"{path.name} (no weights captured)")
    folder = Path(tempfile.mkdtemp(prefix="sba_weights_"))
    try:
        weights = folder / "weights.pt"
        sandbox = SubprocessSandbox(SandboxConfig(timeout_seconds=timeout))
        result = sandbox.run_measured(path.read_text(encoding="utf-8"), max_steps=max_steps, timeout=timeout,
                                      script_args=list(script_args or []), as_file=str(path),
                                      save_weights=str(weights))
        run = result.run_report or {}
        steps = int(run.get("steps") or 0)
        if weights.is_file():
            models = ", ".join(run.get("weights_models") or []) or "model"
            report = inspector.inspect_checkpoint(weights)
            report.model_name = path.stem
            report.source = f"{models} in {path.name} after {steps} training step(s)"
        elif result.timed_out:
            report.diagnostics.append(f"The script did not finish {max_steps} steps within {timeout:g} seconds.")
        else:
            report.diagnostics.append("No weights were captured: " + str(
                run.get("weights_note") or result.error_message or "the script made no optimizer step"))
        if run.get("status") == "error":
            report.diagnostics.append("The script raised an error while running: " + str(run.get("error"))[-300:])
        return report
    finally:
        shutil.rmtree(folder, ignore_errors=True)
