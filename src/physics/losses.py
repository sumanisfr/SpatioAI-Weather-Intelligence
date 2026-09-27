"""Composable physically motivated losses for precipitation downscaling."""

from typing import Any, Dict, Mapping, Optional

import torch

from src.physics.conservation import coarse_consistency_loss, mass_conservation_loss
from src.physics.gradients import gradient_structure_loss
from src.physics.precipitation import extreme_structure_loss, non_negative_loss


def physics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    coarse_input: torch.Tensor,
    coordinates: Optional[Mapping[str, torch.Tensor]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return weighted physical-space loss and transparent components."""
    config = config or {}
    weights = config.get("weights", {})
    percentile = float(config.get("extreme_percentile", 95.0))
    coordinates = coordinates or {}
    high_latitudes = coordinates.get("high_lats")
    coarse_latitudes = coordinates.get("coarse_lats")
    components = {
        "non_negative": non_negative_loss(prediction),
        "coarse_consistency": coarse_consistency_loss(prediction, coarse_input, high_latitudes),
        "mass_conservation": mass_conservation_loss(prediction, coarse_input, high_latitudes, coarse_latitudes),
        "gradient": gradient_structure_loss(prediction, target),
        "extreme_structure": extreme_structure_loss(prediction, target, percentile),
    }
    total = sum(components[name] * float(weights.get(name, 1.0)) for name in components)
    return total, components
