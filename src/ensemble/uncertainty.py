"""Conditional ensemble spread diagnostics."""

from typing import Dict

import torch


def uncertainty_map(statistics: Dict[str, torch.Tensor], normalized: bool = False, epsilon: float = 1e-6) -> torch.Tensor:
    """Return conditional ensemble standard deviation or coefficient of variation."""
    if "std" not in statistics or "mean" not in statistics:
        raise KeyError("statistics must contain mean and std")
    if normalized:
        return statistics["std"] / (statistics["mean"].abs() + epsilon)
    return statistics["std"]
