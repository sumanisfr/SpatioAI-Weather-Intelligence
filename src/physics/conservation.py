"""Area-aware coarse consistency and precipitation conservation utilities."""

from typing import Optional, Tuple

import torch
from torch.nn import functional as F

EARTH_RADIUS_M = 6_371_000.0


def latitude_area_weights(latitudes: torch.Tensor) -> torch.Tensor:
    """Return relative cell-area weights proportional to cos(latitude).

    For equal angular grid spacing, this is sufficient for normalized losses;
    absolute cell widths cancel in the normalized area-weighted mean.
    """
    latitudes = torch.as_tensor(latitudes, dtype=torch.float32)
    if latitudes.ndim != 1:
        raise ValueError("latitudes must be one-dimensional")
    return torch.cos(torch.deg2rad(latitudes)).clamp_min(1e-6)


def _row_weights(latitudes: Optional[torch.Tensor], height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if latitudes is None:
        return torch.ones(height, device=device, dtype=dtype)
    return latitude_area_weights(latitudes.to(device=device, dtype=dtype))


def area_weighted_mean(field: torch.Tensor, latitudes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute an area-weighted spatial mean for [B,C,H,W] fields."""
    weights = _row_weights(latitudes, field.shape[-2], field.device, field.dtype).view(1, 1, -1, 1)
    return (field * weights).sum(dim=(-2, -1)) / (weights.sum() * field.shape[-1]).clamp_min(1e-12)


def aggregate_to_coarse(field: torch.Tensor, coarse_shape: Tuple[int, int], latitudes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Aggregate intensity by area-weighted pooling to a coarse grid.

    This treats precipitation as an interval-mean intensity, so averaging is
    appropriate. It is not a claim of atmospheric mass or water-budget closure.
    """
    if field.ndim != 4:
        raise ValueError("field must have shape [batch, channels, height, width]")
    if latitudes is None:
        return F.adaptive_avg_pool2d(field, coarse_shape)
    weights = _row_weights(latitudes, field.shape[-2], field.device, field.dtype).view(1, 1, -1, 1)
    weighted = field * weights
    pooled = F.adaptive_avg_pool2d(weighted, coarse_shape)
    weight_grid = F.adaptive_avg_pool2d(weights.expand(field.shape[0], field.shape[1], -1, field.shape[-1]), coarse_shape)
    return pooled / weight_grid.clamp_min(1e-12)


def coarse_consistency_loss(
    prediction: torch.Tensor,
    coarse_input: torch.Tensor,
    prediction_latitudes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compare area-averaged high-resolution intensity with coarse input."""
    reconstructed = aggregate_to_coarse(prediction, coarse_input.shape[-2:], prediction_latitudes)
    scale = coarse_input.detach().abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (reconstructed - coarse_input).abs().div(scale).mean()


def mass_conservation_loss(
    prediction: torch.Tensor,
    coarse_input: torch.Tensor,
    prediction_latitudes: Optional[torch.Tensor] = None,
    coarse_latitudes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compare domain-integrated precipitation intensity, normalized by coarse total.

    The conserved proxy is area-integrated precipitation intensity over the same
    crop and timestep, not atmospheric mass. Latitude-dependent relative areas
    are used because geographic grid cells shrink toward the poles.
    """
    high_total = area_weighted_mean(prediction, prediction_latitudes)
    coarse_total = area_weighted_mean(coarse_input, coarse_latitudes).detach()
    return ((high_total - coarse_total) / coarse_total.abs().clamp_min(1.0)).abs().mean()
