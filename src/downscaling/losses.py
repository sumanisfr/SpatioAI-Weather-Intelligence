"""
Loss functions for extreme-event weather downscaling.
Supports MSE, MAE, Weighted MSE (with high weights for extreme percentiles), and Gradient Loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedExtremeMSELoss(nn.Module):
    """
    Weighted Mean Squared Error giving extra penalty to extreme heavy precipitation events.
    Loss = Mean( (1 + λ * I(target > thresh)) * (pred - target)^2 )
    """

    def __init__(
        self,
        extreme_percentile: float = 95.0,
        extreme_weight: float = 3.0,
        fixed_threshold: Optional[float] = None,
    ):
        """
        Args:
            extreme_percentile: Percentile threshold in target batch to apply higher weight.
            extreme_weight: Multiplier weight for values exceeding the threshold.
            fixed_threshold: Optional fixed physical threshold in mm (overrides percentile if set).
        """
        super().__init__()
        self.extreme_percentile = extreme_percentile
        self.extreme_weight = extreme_weight
        self.fixed_threshold = fixed_threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_sq = (pred - target) ** 2

        if self.fixed_threshold is not None:
            thresh = torch.tensor(self.fixed_threshold, device=target.device, dtype=target.dtype)
        else:
            # Batch-level percentile
            if target.numel() > 0:
                k = max(1, int((1.0 - self.extreme_percentile / 100.0) * target.numel()))
                # Top-k threshold approximation for differentiability
                thresh = torch.topk(target.view(-1), k=k).values[-1].detach()
            else:
                thresh = torch.tensor(0.0, device=target.device)

        # Weight mask: 1.0 for normal, extreme_weight for extreme
        weights = 1.0 + (self.extreme_weight - 1.0) * (target > thresh).float()
        weighted_loss = torch.mean(weights * diff_sq)
        return weighted_loss


class SpatialGradientLoss(nn.Module):
    """
    Spatial gradient / total-variation loss to discourage overly blurry or smoothed predictions.
    Computes L1 difference between spatial gradients of prediction and target.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Spatial gradients along height (dy) and width (dx)
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]

        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        loss_dy = F.l1_loss(pred_dy, target_dy)
        loss_dx = F.l1_loss(pred_dx, target_dx)
        return loss_dy + loss_dx


class CombinedDownscalingLoss(nn.Module):
    """
    Combines reconstruction loss (MSE/MAE), extreme-weighted loss, and spatial gradient loss.
    """

    def __init__(
        self,
        base_loss: str = "weighted_mse",
        extreme_percentile: float = 95.0,
        extreme_weight: float = 3.0,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss_name = base_loss
        self.gradient_weight = gradient_weight
        self.grad_loss = SpatialGradientLoss() if gradient_weight > 0 else None

        if base_loss == "mse":
            self.recon_loss = nn.MSELoss()
        elif base_loss == "mae":
            self.recon_loss = nn.L1Loss()
        elif base_loss == "weighted_mse":
            self.recon_loss = WeightedExtremeMSELoss(
                extreme_percentile=extreme_percentile,
                extreme_weight=extreme_weight,
            )
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.recon_loss(pred, target)
        if self.grad_loss is not None:
            loss = loss + self.gradient_weight * self.grad_loss(pred, target)
        return loss


def get_downscaling_loss(
    loss_name: str = "weighted_mse",
    extreme_percentile: float = 95.0,
    extreme_weight: float = 3.0,
    gradient_weight: float = 0.05,
) -> nn.Module:
    """Factory helper to instantiate downscaling loss."""
    return CombinedDownscalingLoss(
        base_loss=loss_name,
        extreme_percentile=extreme_percentile,
        extreme_weight=extreme_weight,
        gradient_weight=gradient_weight,
    )
