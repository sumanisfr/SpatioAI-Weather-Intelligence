"""Statistics for conditional precipitation ensembles."""

from typing import Dict, Iterable, Sequence

import numpy as np
import torch


def _as_tensor(samples):
    return samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)


def compute_ensemble_statistics(samples, sample_dim: int = 1, quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)) -> Dict[str, torch.Tensor]:
    """Compute statistics over the conditional-sample dimension.

    Phase 6 sampler output uses ``[batch, samples, channels, height, width]``.
    """
    tensor = _as_tensor(samples)
    if tensor.ndim < 2:
        raise ValueError("samples must have a batch/sample dimension")
    if not 0 <= sample_dim < tensor.ndim:
        raise ValueError("sample_dim is out of range")
    result = {
        "mean": tensor.mean(dim=sample_dim),
        "median": torch.quantile(tensor, 0.50, dim=sample_dim),
        "std": tensor.std(dim=sample_dim, unbiased=False),
    }
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantiles must be between 0 and 1")
        key = f"q{int(round(quantile * 100)):02d}"
        result[key] = torch.quantile(tensor, quantile, dim=sample_dim)
    return result


def statistics_to_numpy(statistics: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert a statistics dictionary to NumPy arrays for serialization."""
    return {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value) for key, value in statistics.items()}
