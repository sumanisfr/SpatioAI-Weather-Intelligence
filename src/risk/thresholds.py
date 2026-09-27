"""Configurable fixed and climatological threshold resolution."""

from typing import Optional

import torch


def resolve_thresholds(
    mode: str,
    fixed_thresholds: Optional[list[float]] = None,
    climatological_field: Optional[torch.Tensor] = None,
    percentiles: Optional[list[float]] = None,
) -> list[float] | torch.Tensor:
    """Resolve fixed thresholds or local climatological percentile fields."""
    mode = mode.lower()
    if mode == "fixed":
        if not fixed_thresholds:
            raise ValueError("fixed threshold mode requires fixed_thresholds")
        return [float(value) for value in fixed_thresholds]
    if mode == "climatological_percentile":
        if climatological_field is None or not percentiles:
            raise ValueError("climatological mode requires a field and percentiles")
        return torch.stack([torch.quantile(climatological_field, float(percentile) / 100.0, dim=0) for percentile in percentiles])
    raise ValueError("mode must be 'fixed' or 'climatological_percentile'")
