"""Weight-Space Perception: Inspector.

Extracts, inspects, and analyzes raw neural network weights across any architecture:
- SVD / Spectral Decomposition (rank collapse, ill-conditioned matrices)
- Heavy-tailed Kurtosis / Outliers (the culprit in reasoning/attention loops)
- Extreme Bias Shifts & Dead Activation Directions
- NaN / Infs and Frozen Parameters
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .schemas import WeightAnomalyType, WeightFinding, WeightReport


class WeightInspector:
    """Natively inspects raw weights and discovers weight-space anomalies."""

    def __init__(
        self,
        *,
        max_condition_number: float = 1e4,
        min_effective_rank_ratio: float = 0.15,
        max_kurtosis: float = 30.0,
        extreme_bias_threshold: float = -15.0,
        svd_max_dim: int = 1024,
    ) -> None:
        self.max_condition_number = max_condition_number
        self.min_effective_rank_ratio = min_effective_rank_ratio
        self.max_kurtosis = max_kurtosis
        self.extreme_bias_threshold = extreme_bias_threshold
        self.svd_max_dim = svd_max_dim

    # -----------------------------------------------------------------------
    # Public Entry Points
    # -----------------------------------------------------------------------

    def inspect_model(self, model: nn.Module, model_name: str = "Model") -> WeightReport:
        """Inspect all parameters of a live PyTorch nn.Module."""
        state_dict = model.state_dict()
        return self.inspect_state_dict(state_dict, model_name=model_name)

    def inspect_checkpoint(self, path: Union[str, Path]) -> WeightReport:
        """Inspect weights from a saved checkpoint file (.pt, .pth, .bin, .safetensors)."""
        file_path = Path(path).resolve()
        if not file_path.is_file():
            report = WeightReport(model_name=file_path.name, total_parameters=0, trainable_parameters=0)
            report.diagnostics.append(f"Checkpoint not found: {file_path}")
            return report

        state_dict: Dict[str, torch.Tensor] = {}
        if file_path.suffix == ".safetensors":
            try:
                import safetensors.torch
                state_dict = safetensors.torch.load_file(str(file_path))
            except ImportError:
                report = WeightReport(model_name=file_path.name, total_parameters=0, trainable_parameters=0)
                report.diagnostics.append("safetensors package required to read .safetensors checkpoints")
                return report
        else:
            try:
                loaded = torch.load(file_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                        state_dict = loaded["state_dict"]
                    elif "model" in loaded and isinstance(loaded["model"], dict):
                        state_dict = loaded["model"]
                    else:
                        state_dict = {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
                elif isinstance(loaded, nn.Module):
                    state_dict = loaded.state_dict()
            except Exception as exc:
                report = WeightReport(model_name=file_path.name, total_parameters=0, trainable_parameters=0)
                report.diagnostics.append(f"Failed to load checkpoint: {exc}")
                return report

        return self.inspect_state_dict(state_dict, model_name=file_path.stem)

    def inspect_state_dict(
        self, state_dict: Dict[str, torch.Tensor], model_name: str = "Model"
    ) -> WeightReport:
        """Inspect a raw dictionary of parameter tensors."""
        findings: List[WeightFinding] = []
        layer_summaries: Dict[str, Dict[str, Any]] = {}
        total_params = 0

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            total_params += tensor.numel()

            # Flatten to CPU float for numerical stability
            t_float = tensor.detach().cpu().to(torch.float32)

            # 1. NaN / Inf Check
            if not torch.isfinite(t_float).all():
                findings.append(
                    WeightFinding(
                        layer_name=name,
                        param_shape=tuple(tensor.shape),
                        anomaly_type=WeightAnomalyType.NAN_INF,
                        severity="critical",
                        metric_value=float(torch.isnan(t_float).sum() + torch.isinf(t_float).sum()),
                        threshold=0.0,
                        message=f"Layer '{name}' contains non-finite (NaN/Inf) values.",
                        suggested_action="Reset or sanitize non-finite parameter entries.",
                    )
                )
                continue

            # 2. Frozen / All-Zero Check
            t_norm = float(torch.norm(t_float).item())
            if t_norm == 0.0 and tensor.numel() > 1:
                findings.append(
                    WeightFinding(
                        layer_name=name,
                        param_shape=tuple(tensor.shape),
                        anomaly_type=WeightAnomalyType.FROZEN_OR_ZERO,
                        severity="warning",
                        metric_value=0.0,
                        threshold=1e-8,
                        message=f"Layer '{name}' is completely zeroed out (norm = 0.0).",
                        suggested_action="Re-initialize weights or check for vanishing gradient collapse.",
                    )
                )

            # 3. Bias Vector Analysis
            if name.endswith("bias") and tensor.dim() == 1:
                bias_finding = self._check_bias(name, t_float)
                if bias_finding:
                    findings.append(bias_finding)

            # 4. 2D / Multi-dim Weight Matrix Analysis
            if tensor.dim() >= 2:
                matrix_findings, summary = self._check_matrix(name, t_float)
                findings.extend(matrix_findings)
                if summary:
                    layer_summaries[name] = summary

        return WeightReport(
            model_name=model_name,
            total_parameters=total_params,
            trainable_parameters=total_params,
            findings=findings,
            layer_summaries=layer_summaries,
        )

    # -----------------------------------------------------------------------
    # Analytical Sub-methods
    # -----------------------------------------------------------------------

    def _check_bias(self, name: str, bias: torch.Tensor) -> Optional[WeightFinding]:
        """Inspect bias parameters for activation collapse or extreme saturation."""
        min_bias = float(bias.min().item())
        max_bias = float(bias.max().item())
        mean_bias = float(bias.mean().item())

        # Check for catastrophic negative bias (e.g. benchmark_06 dead_relu pattern: bias = -100.0)
        if min_bias <= self.extreme_bias_threshold:
            pct_dead = float((bias <= self.extreme_bias_threshold).sum().item()) / bias.numel() * 100.0
            return WeightFinding(
                layer_name=name,
                param_shape=tuple(bias.shape),
                anomaly_type=WeightAnomalyType.BIAS_COLLAPSE,
                severity="critical",
                metric_value=min_bias,
                threshold=self.extreme_bias_threshold,
                message=(
                    f"Catastrophic negative bias in '{name}' (min = {min_bias:.2f}). "
                    f"{pct_dead:.1f}% of units will be completely dead under ReLU."
                ),
                evidence={
                    "min_bias": min_bias,
                    "max_bias": max_bias,
                    "mean_bias": mean_bias,
                    "pct_dead": pct_dead,
                },
                suggested_action="Surgically re-center bias vector to zero (bias.zero_()).",
            )
        return None

    def _check_matrix(
        self, name: str, tensor: torch.Tensor
    ) -> Tuple[List[WeightFinding], Optional[Dict[str, Any]]]:
        """Inspect weight matrices for rank collapse, ill-conditioning, and kurtosis outliers."""
        findings: List[WeightFinding] = []

        # Reshape conv or higher-dim weights to 2D [out_features, in_features]
        if tensor.dim() > 2:
            matrix = tensor.view(tensor.shape[0], -1)
        else:
            matrix = tensor

        rows, cols = matrix.shape
        min_dim = min(rows, cols)
        if min_dim < 2:
            return findings, None

        mean = float(matrix.mean().item())
        std = float(matrix.std().item())
        frob_norm = float(torch.norm(matrix).item())

        # Kurtosis / Outlier check (heavy-tailed distributions cause attention/reasoning loops)
        if std > 1e-9 and matrix.numel() >= 64:
            centered = matrix - mean
            m4 = torch.mean(centered**4).item()
            m2 = torch.mean(centered**2).item()
            kurtosis = float(m4 / (m2**2 + 1e-12)) if m2 > 0 else 0.0

            if kurtosis > self.max_kurtosis:
                max_entry = float(torch.max(torch.abs(matrix)).item())
                findings.append(
                    WeightFinding(
                        layer_name=name,
                        param_shape=tuple(tensor.shape),
                        anomaly_type=WeightAnomalyType.WEIGHT_OUTLIER,
                        severity="warning",
                        metric_value=kurtosis,
                        threshold=self.max_kurtosis,
                        message=(
                            f"Heavy-tailed weight outliers in '{name}' (kurtosis = {kurtosis:.1f} > {self.max_kurtosis}). "
                            f"Extreme outlier entry = {max_entry:.3f} ({max_entry / std:.1f}x std)."
                        ),
                        evidence={"kurtosis": kurtosis, "max_val": max_entry, "std": std},
                        suggested_action="Surgically clamp or scale high-kurtosis outlier entries.",
                    )
                )

        # SVD / Spectral Analysis
        # Limit dimension for fast inspection
        if max(rows, cols) <= self.svd_max_dim:
            try:
                s = torch.linalg.svdvals(matrix)
                s_np = s.numpy()
                sigma_max = float(s_np[0])
                sigma_min = float(s_np[-1]) if s_np[-1] > 1e-12 else 1e-12

                # Condition Number
                condition_number = sigma_max / sigma_min

                # Effective Rank (exponential of spectral entropy)
                s_sum = float(s.sum().item())
                if s_sum > 0:
                    probs = s / s_sum
                    entropy = -float(torch.sum(probs * torch.log(probs + 1e-12)).item())
                    eff_rank = float(math.exp(entropy))
                else:
                    eff_rank = 0.0

                rank_ratio = eff_rank / min_dim

                summary = {
                    "shape": list(matrix.shape),
                    "frob_norm": round(frob_norm, 4),
                    "spectral_norm": round(sigma_max, 4),
                    "condition_number": round(condition_number, 2),
                    "effective_rank": round(eff_rank, 2),
                    "rank_ratio": round(rank_ratio, 4),
                }

                # Check Condition Number
                if condition_number > self.max_condition_number:
                    findings.append(
                        WeightFinding(
                            layer_name=name,
                            param_shape=tuple(tensor.shape),
                            anomaly_type=WeightAnomalyType.ILL_CONDITIONED,
                            severity="warning",
                            metric_value=condition_number,
                            threshold=self.max_condition_number,
                            message=(
                                f"Ill-conditioned weight matrix in '{name}' (kappa = {condition_number:.1e}). "
                                "High sensitivity to input perturbations."
                            ),
                            evidence=summary,
                            suggested_action="Apply spectral regularization or singular value scaling.",
                        )
                    )

                # Check Rank Collapse
                if rank_ratio < self.min_effective_rank_ratio and min_dim >= 8:
                    findings.append(
                        WeightFinding(
                            layer_name=name,
                            param_shape=tuple(tensor.shape),
                            anomaly_type=WeightAnomalyType.RANK_COLLAPSE,
                            severity="critical" if rank_ratio < 0.05 else "warning",
                            metric_value=rank_ratio,
                            threshold=self.min_effective_rank_ratio,
                            message=(
                                f"Severe rank collapse in '{name}': effective rank is {eff_rank:.1f} / {min_dim} "
                                f"({rank_ratio * 100:.1f}% capacity utilization)."
                            ),
                            evidence=summary,
                            suggested_action="Inject low-rank delta or re-orthogonalize collapsed subspace.",
                        )
                    )

                return findings, summary
            except Exception:
                pass

        return findings, None

