"""Differentiable spatial gradient diagnostics."""

from typing import Tuple

import torch


def spatial_gradient(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return finite differences in row and column directions."""
    if field.ndim != 4:
        raise ValueError("field must have shape [batch, channels, height, width]")
    dy = torch.zeros_like(field)
    dx = torch.zeros_like(field)
    dy[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) * 0.5
    dy[..., 0, :] = field[..., 1, :] - field[..., 0, :]
    dy[..., -1, :] = field[..., -1, :] - field[..., -2, :]
    dx[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) * 0.5
    dx[..., :, 0] = field[..., :, 1] - field[..., :, 0]
    dx[..., :, -1] = field[..., :, -1] - field[..., :, -2]
    return dy, dx


def gradient_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Return differentiable Euclidean gradient magnitude."""
    dy, dx = spatial_gradient(field)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def gradient_structure_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare gradient structure instead of forcing gradients toward zero."""
    target_gradient = gradient_magnitude(target).detach()
    scale = target_gradient.mean().clamp_min(1.0)
    return torch.mean(torch.abs(gradient_magnitude(prediction) - target_gradient)) / scale
