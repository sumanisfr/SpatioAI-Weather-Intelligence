"""Differentiable precipitation constraints in physical units."""

from typing import Optional

import torch
from torch.nn import functional as F


def non_negative_loss(prediction: torch.Tensor) -> torch.Tensor:
    """Penalize negative precipitation without clipping the training signal."""
    return F.relu(-prediction).square().mean()


def precipitation_extreme_threshold(field: torch.Tensor, percentile: float = 95.0) -> torch.Tensor:
    """Return one detached per-batch threshold for smooth extreme masks."""
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between 0 and 100")
    return torch.quantile(field.detach().flatten(1), percentile / 100.0, dim=1, keepdim=True)


def smooth_extreme_mask(field: torch.Tensor, percentile: float = 95.0, temperature: Optional[float] = None) -> torch.Tensor:
    """Create a differentiable approximation to a percentile exceedance mask."""
    threshold = precipitation_extreme_threshold(field, percentile).view(-1, 1, 1, 1)
    scale = temperature if temperature is not None else max(float(field.detach().abs().mean()), 1.0) * 0.05
    return torch.sigmoid((field - threshold) / max(scale, 1e-6))


def extreme_structure_loss(prediction: torch.Tensor, target: torch.Tensor, percentile: float = 95.0) -> torch.Tensor:
    """Match smooth extreme area and intensity-weighted extreme structure."""
    pred_mask = smooth_extreme_mask(prediction, percentile)
    target_mask = smooth_extreme_mask(target, percentile).detach()
    area_loss = (pred_mask.mean(dim=(-2, -1)) - target_mask.mean(dim=(-2, -1))).abs().mean()
    pred_weighted = (pred_mask * prediction.clamp_min(0.0)).mean(dim=(-2, -1))
    target_weighted = (target_mask * target.clamp_min(0.0)).mean(dim=(-2, -1)).detach()
    intensity_scale = target_weighted.detach().abs().mean().clamp_min(1.0)
    intensity_loss = F.smooth_l1_loss(pred_weighted / intensity_scale, target_weighted / intensity_scale)
    return area_loss + intensity_loss
