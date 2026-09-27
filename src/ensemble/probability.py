"""Empirical threshold-exceedance diagnostics."""

from typing import Sequence, Union

import torch


def threshold_exceedance_probability(samples: torch.Tensor, threshold: Union[float, torch.Tensor], sample_dim: int = 1) -> torch.Tensor:
    """Estimate empirical frequency of samples exceeding a physical threshold."""
    if samples.ndim < 2:
        raise ValueError("samples must have a sample dimension")
    return (samples > threshold).to(dtype=samples.dtype).mean(dim=sample_dim)


def multiple_threshold_probabilities(samples: torch.Tensor, thresholds: Sequence[float], sample_dim: int = 1) -> dict[str, torch.Tensor]:
    """Compute empirical exceedance fields for several configurable thresholds."""
    return {str(threshold): threshold_exceedance_probability(samples, threshold, sample_dim) for threshold in thresholds}
