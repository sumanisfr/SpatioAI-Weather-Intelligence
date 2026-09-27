"""Explainable event-level risk descriptors."""

import torch


def transparent_risk_score(
    exceedance_probability: torch.Tensor,
    severity: torch.Tensor,
    affected_area_km2: torch.Tensor,
    area_scale_km2: float = 1000.0,
) -> torch.Tensor:
    """Compute ``probability * severity * normalized affected area``.

    This is an experimental descriptive score, not an official warning level.
    """
    area_factor = affected_area_km2 / max(float(area_scale_km2), 1e-6)
    return exceedance_probability * severity * area_factor
