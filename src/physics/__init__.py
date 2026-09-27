"""Physically motivated, differentiable constraints for Phase 7."""

from src.physics.conservation import (
    aggregate_to_coarse,
    area_weighted_mean,
    coarse_consistency_loss,
    latitude_area_weights,
    mass_conservation_loss,
)
from src.physics.gradients import gradient_magnitude, gradient_structure_loss, spatial_gradient
from src.physics.losses import physics_loss
from src.physics.moisture import moisture_proxy_loss
from src.physics.precipitation import extreme_structure_loss, non_negative_loss, smooth_extreme_mask

__all__ = [
    "latitude_area_weights",
    "area_weighted_mean",
    "aggregate_to_coarse",
    "coarse_consistency_loss",
    "mass_conservation_loss",
    "spatial_gradient",
    "gradient_magnitude",
    "gradient_structure_loss",
    "non_negative_loss",
    "smooth_extreme_mask",
    "extreme_structure_loss",
    "moisture_proxy_loss",
    "physics_loss",
]
