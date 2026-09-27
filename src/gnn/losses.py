"""
Loss functions and optimization objectives for SpatioAI GNN (Phase 4).
Handles severe edge class imbalance (few positive associations vs many negative candidate pairs).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance in link/association prediction.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(
    loss_type: str = "bce_weighted",
    pos_weight: float = 3.0,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function to instantiate configurable GNN loss functions.

    Args:
        loss_type: "bce_weighted", "bce", or "focal".
        pos_weight: Positive class weight multiplier for BCEWithLogitsLoss.
        focal_alpha: Class weighting for FocalLoss.
        focal_gamma: Focusing parameter for FocalLoss.

    Returns:
        nn.Module loss instance.
    """
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "bce_weighted":
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Choose 'bce_weighted', 'bce', or 'focal'.")
