"""Sampling and ensemble summaries for conditional diffusion downscaling."""

from typing import Dict, Optional

import torch

from src.downscaling.diffusion import DiffusionProcess, inverse_log1p_transform, sample_diffusion


def generate(
    model: torch.nn.Module,
    process: DiffusionProcess,
    condition: torch.Tensor,
    num_samples: int = 1,
    seed: Optional[int] = None,
    transformed: bool = False,
) -> torch.Tensor:
    """Generate physical precipitation samples, optionally retaining log space."""
    samples = sample_diffusion(model, process, condition, num_samples=num_samples, seed=seed)
    return samples if transformed else inverse_log1p_transform(samples)


def summarize_ensemble(samples: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return ensemble mean, spread, and common quantiles along sample dimension."""
    if samples.ndim < 2:
        raise ValueError("samples must include a sample dimension at axis 1")
    return {
        "mean": samples.mean(dim=1),
        "std": samples.std(dim=1, unbiased=False),
        "q10": torch.quantile(samples, 0.10, dim=1),
        "q50": torch.quantile(samples, 0.50, dim=1),
        "q90": torch.quantile(samples, 0.90, dim=1),
    }


__all__ = ["generate", "summarize_ensemble"]
