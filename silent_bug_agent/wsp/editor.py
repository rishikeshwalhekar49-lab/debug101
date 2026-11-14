"""Weight-Space Surgery: Editor.

Generates and applies surgical weight-space modifications directly:
- Subspace Excision: W' = W(I - v v^T) to eliminate failure/loop directions
- Low-Rank Delta Injection: W' = W + alpha * (u v^T)
- Bias Centering & Dead Neuron Resets
- Outlier Clamping for high-kurtosis reasoning-loop excisions
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .schemas import WeightAnomalyType, WeightFinding, WeightPatch, WeightSurgeryType


class WeightEditor:
    """Performs surgical, gradient-free modifications directly in neural network weight space."""

    def __init__(self, *, default_alpha: float = 0.05, outlier_sigma_threshold: float = 4.0) -> None:
        self.default_alpha = default_alpha
        self.outlier_sigma_threshold = outlier_sigma_threshold

    # -----------------------------------------------------------------------
    # Patch Synthesis from Weight Findings
    # -----------------------------------------------------------------------

    def synthesize_patch(self, finding: WeightFinding, tensor: torch.Tensor) -> Optional[WeightPatch]:
        """Automatically synthesize a surgical WeightPatch to resolve a specific WeightFinding."""
        if finding.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE:
            return self.create_bias_center_patch(finding.layer_name, tensor)

        if finding.anomaly_type == WeightAnomalyType.WEIGHT_OUTLIER:
            return self.create_outlier_clamp_patch(finding.layer_name, tensor)

        if finding.anomaly_type == WeightAnomalyType.RANK_COLLAPSE:
            return self.create_rank_restoration_patch(finding.layer_name, tensor)

        if finding.anomaly_type == WeightAnomalyType.ILL_CONDITIONED:
            return self.create_spectral_scale_patch(finding.layer_name, tensor)

        if finding.anomaly_type == WeightAnomalyType.DEAD_DIRECTION:
            return self.create_subspace_excision_patch(finding.layer_name, tensor)

        return None

    # -----------------------------------------------------------------------
    # Specialized Surgery Constructors
    # -----------------------------------------------------------------------

    def create_bias_center_patch(
        self, layer_name: str, bias: torch.Tensor, target_value: float = 0.0
    ) -> WeightPatch:
        """Surgically re-centers dead or heavily biased bias vectors."""
        orig_norm = float(torch.norm(bias.float()).item())
        affected_mask = bias < -5.0  # elements severely depressed

        def transform(t: torch.Tensor) -> torch.Tensor:
            new_t = t.clone()
            # Reset collapsed negative bias elements
            new_t[affected_mask] = target_value
            return new_t

        new_bias = transform(bias)
        new_norm = float(torch.norm(new_bias.float()).item())
        delta_norm = float(torch.norm((new_bias - bias).float()).item())
        num_affected = int(affected_mask.sum().item())
        pct_affected = (num_affected / max(1, bias.numel())) * 100.0

        return WeightPatch(
            target_param=layer_name,
            surgery_type=WeightSurgeryType.BIAS_CENTER,
            description=f"Re-center {num_affected} collapsed bias entries ({pct_affected:.1f}%) to {target_value}",
            params_affected=num_affected,
            pct_affected=pct_affected,
            original_norm=orig_norm,
            new_norm=new_norm,
            delta_norm=delta_norm,
            meta={"target_value": target_value, "num_affected": num_affected},
            transform_fn=transform,
        )

    def create_subspace_excision_patch(
        self, layer_name: str, matrix: torch.Tensor, direction_v: Optional[torch.Tensor] = None
    ) -> WeightPatch:
        """Surgically excises a toxic or loop direction: W' = W(I - v v^T)."""
        orig_norm = float(torch.norm(matrix.float()).item())
        is_high_dim = matrix.dim() > 2
        flat_matrix = matrix.view(matrix.shape[0], -1) if is_high_dim else matrix
        rows, cols = flat_matrix.shape

        if direction_v is None:
            # Find the first principal right singular vector
            try:
                _, _, Vh = torch.linalg.svd(flat_matrix.float(), full_matrices=False)
                v = Vh[0:1, :].T  # [cols, 1]
            except Exception:
                v = torch.randn(cols, 1)
                v = v / (torch.norm(v) + 1e-12)
        else:
            v = direction_v.float().view(cols, 1)
            v = v / (torch.norm(v) + 1e-12)

        def transform(t: torch.Tensor) -> torch.Tensor:
            shape = t.shape
            w_2d = t.view(t.shape[0], -1).float()
            # Subspace projection subtraction: W' = W - (W v) v^T
            proj = torch.matmul(w_2d, v)  # [rows, 1]
            w_excised = w_2d - torch.matmul(proj, v.T)
            return w_excised.view(shape).to(t.dtype)

        new_t = transform(matrix)
        new_norm = float(torch.norm(new_t.float()).item())
        delta_norm = float(torch.norm((new_t - matrix).float()).item())
        pct = (cols / max(1, matrix.numel())) * 100.0

        return WeightPatch(
            target_param=layer_name,
            surgery_type=WeightSurgeryType.SUBSPACE_EXCISION,
            description=f"Excise dominant failure subspace direction v (subspace projection subtraction)",
            params_affected=matrix.numel(),
            pct_affected=pct,
            original_norm=orig_norm,
            new_norm=new_norm,
            delta_norm=delta_norm,
            meta={"projection_dim": 1},
            transform_fn=transform,
        )

    def create_outlier_clamp_patch(
        self, layer_name: str, matrix: torch.Tensor, sigma_factor: Optional[float] = None
    ) -> WeightPatch:
        """Surgically clamps heavy-tailed outlier entries responsible for reasoning loops."""
        orig_norm = float(torch.norm(matrix.float()).item())
        mean = float(matrix.float().mean().item())
        std = float(matrix.float().std().item())
        factor = sigma_factor or self.outlier_sigma_threshold
        threshold = factor * std

        outliers_mask = torch.abs(matrix.float() - mean) > threshold
        num_outliers = int(outliers_mask.sum().item())
        pct_affected = (num_outliers / max(1, matrix.numel())) * 100.0

        def transform(t: torch.Tensor) -> torch.Tensor:
            new_t = t.clone().float()
            clamped = torch.clamp(new_t, min=mean - threshold, max=mean + threshold)
            return clamped.to(t.dtype)

        new_t = transform(matrix)
        new_norm = float(torch.norm(new_t.float()).item())
        delta_norm = float(torch.norm((new_t - matrix).float()).item())

        return WeightPatch(
            target_param=layer_name,
            surgery_type=WeightSurgeryType.OUTLIER_CLAMP,
            description=f"Clamp {num_outliers} outlier weights ({pct_affected:.3f}%) exceeding {factor:.1f} sigma",
            params_affected=num_outliers,
            pct_affected=pct_affected,
            original_norm=orig_norm,
            new_norm=new_norm,
            delta_norm=delta_norm,
            meta={"num_outliers": num_outliers, "sigma_threshold": threshold},
            transform_fn=transform,
        )

    def create_rank_restoration_patch(
        self, layer_name: str, matrix: torch.Tensor, rank_k: int = 2
    ) -> WeightPatch:
        """Injects a low-rank orthogonal delta W' = W + alpha * (u v^T) to revive collapsed rank."""
        orig_norm = float(torch.norm(matrix.float()).item())
        is_high_dim = matrix.dim() > 2
        flat_matrix = matrix.view(matrix.shape[0], -1) if is_high_dim else matrix
        rows, cols = flat_matrix.shape

        # Construct random orthogonal low-rank delta
        u, _ = torch.linalg.qr(torch.randn(rows, rank_k))
        v, _ = torch.linalg.qr(torch.randn(cols, rank_k))
        scale = self.default_alpha * (orig_norm / math.sqrt(rows * cols + 1e-12))

        def transform(t: torch.Tensor) -> torch.Tensor:
            shape = t.shape
            w_2d = t.view(t.shape[0], -1).float()
            delta = scale * torch.matmul(u, v.T)
            return (w_2d + delta).view(shape).to(t.dtype)

        new_t = transform(matrix)
        new_norm = float(torch.norm(new_t.float()).item())
        delta_norm = float(torch.norm((new_t - matrix).float()).item())

        return WeightPatch(
            target_param=layer_name,
            surgery_type=WeightSurgeryType.LOW_RANK_DELTA,
            description=f"Inject rank-{rank_k} orthogonal delta to restore collapsed capacity",
            params_affected=matrix.numel(),
            pct_affected=100.0,
            original_norm=orig_norm,
            new_norm=new_norm,
            delta_norm=delta_norm,
            meta={"rank": rank_k, "alpha": scale},
            transform_fn=transform,
        )

    def create_spectral_scale_patch(
        self, layer_name: str, matrix: torch.Tensor, target_condition: float = 500.0
    ) -> WeightPatch:
        """Surgically regularizes singular value decay to stabilize an ill-conditioned matrix."""
        orig_norm = float(torch.norm(matrix.float()).item())
        is_high_dim = matrix.dim() > 2
        flat_matrix = matrix.view(matrix.shape[0], -1) if is_high_dim else matrix
        rows, cols = flat_matrix.shape

        def transform(t: torch.Tensor) -> torch.Tensor:
            shape = t.shape
            w_2d = t.view(t.shape[0], -1).float()
            try:
                U, S, Vh = torch.linalg.svd(w_2d, full_matrices=False)
                # Soft floor on the smallest singular values to bound condition number
                sigma_max = S[0]
                floor = sigma_max / target_condition
                S_regularized = torch.clamp(S, min=floor)
                w_reg = torch.matmul(U, torch.matmul(torch.diag(S_regularized), Vh))
                return w_reg.view(shape).to(t.dtype)
            except Exception:
                return t

        new_t = transform(matrix)
        new_norm = float(torch.norm(new_t.float()).item())
        delta_norm = float(torch.norm((new_t - matrix).float()).item())

        return WeightPatch(
            target_param=layer_name,
            surgery_type=WeightSurgeryType.SPECTRAL_SCALE,
            description=f"Bound condition number to kappa <= {target_condition:.0f} via spectral regularization",
            params_affected=matrix.numel(),
            pct_affected=100.0,
            original_norm=orig_norm,
            new_norm=new_norm,
            delta_norm=delta_norm,
            meta={"target_condition": target_condition},
            transform_fn=transform,
        )

    # -----------------------------------------------------------------------
    # Application of Patches
    # -----------------------------------------------------------------------

    def apply_patch(
        self, target: Union[nn.Module, Dict[str, torch.Tensor]], patch: WeightPatch
    ) -> bool:
        """Apply a WeightPatch to a live PyTorch model or state_dict in-place."""
        if patch.transform_fn is None:
            return False

        if isinstance(target, nn.Module):
            state_dict = target.state_dict()
            if patch.target_param in state_dict:
                old_tensor = state_dict[patch.target_param]
                new_tensor = patch.transform_fn(old_tensor)
                state_dict[patch.target_param] = new_tensor
                target.load_state_dict(state_dict)
                return True
            # Check named_parameters
            for name, param in target.named_parameters():
                if name == patch.target_param:
                    with torch.no_grad():
                        param.copy_(patch.transform_fn(param))
                    return True
        elif isinstance(target, dict):
            if patch.target_param in target:
                target[patch.target_param] = patch.transform_fn(target[patch.target_param])
                return True

        return False

