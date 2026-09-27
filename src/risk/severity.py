"""Transparent precipitation severity descriptors."""

import torch


def severity_index(intensity: torch.Tensor, threshold, cap: float | None = None) -> torch.Tensor:
    """Return intensity divided by a reference threshold.

    Values are descriptive ratios, not official warning categories.
    """
    severity = intensity / torch.as_tensor(threshold, dtype=intensity.dtype, device=intensity.device).clamp_min(1e-6)
    return severity.clamp_max(cap) if cap is not None else severity
