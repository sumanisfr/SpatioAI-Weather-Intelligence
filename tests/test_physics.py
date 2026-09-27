"""Deterministic tests for Phase 7 physically motivated constraints."""

import pytest
import torch

from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess, inverse_log1p_transform, log1p_transform
from src.physics import (
    aggregate_to_coarse,
    coarse_consistency_loss,
    extreme_structure_loss,
    gradient_magnitude,
    gradient_structure_loss,
    latitude_area_weights,
    mass_conservation_loss,
    non_negative_loss,
    physics_loss,
    spatial_gradient,
)


def test_latitude_area_weights_decrease_toward_pole():
    weights = latitude_area_weights(torch.tensor([0.0, 30.0, 60.0]))
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert weights[0] > weights[-1]


def test_gradient_known_field():
    field = torch.arange(16.0).reshape(1, 1, 4, 4)
    dy, dx = spatial_gradient(field)
    assert dy.shape == field.shape and dx.shape == field.shape
    assert torch.allclose(dx, torch.ones_like(dx))
    assert torch.allclose(dy, torch.full_like(dy, 4.0))
    assert torch.isfinite(gradient_magnitude(field)).all()


def test_coarse_consistency_exact_and_mismatch():
    high = torch.ones(1, 1, 8, 8)
    coarse = torch.ones(1, 1, 2, 2)
    assert coarse_consistency_loss(high, coarse) == pytest.approx(0.0, abs=1e-6)
    assert coarse_consistency_loss(high * 3.0, coarse) > 0.1
    assert aggregate_to_coarse(high, (2, 2)).shape == coarse.shape


def test_mass_conservation_exact_and_mismatch():
    high = torch.ones(1, 1, 8, 8)
    coarse = torch.ones(1, 1, 2, 2)
    lat_high = torch.linspace(0.0, 60.0, 8)
    lat_coarse = torch.linspace(0.0, 60.0, 2)
    assert mass_conservation_loss(high, coarse, lat_high, lat_coarse) < 1e-6
    assert mass_conservation_loss(high * 2.0, coarse, lat_high, lat_coarse) > 0.1


def test_non_negative_and_extreme_structure_losses():
    positive = torch.ones(1, 1, 8, 8)
    negative = -torch.ones_like(positive)
    assert non_negative_loss(positive) == pytest.approx(0.0)
    assert non_negative_loss(negative) > 0
    assert extreme_structure_loss(positive, positive) < 1e-6
    assert gradient_structure_loss(positive, positive) < 1e-6


def test_combined_loss_is_differentiable():
    prediction = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8)
    coarse = torch.rand(1, 1, 2, 2)
    total, components = physics_loss(
        prediction,
        target,
        coarse,
        {"high_lats": torch.linspace(0.0, 60.0, 8), "coarse_lats": torch.linspace(0.0, 60.0, 2)},
        {"weights": {name: 1.0 for name in components_names()}},
    )
    assert set(components) == set(components_names())
    assert torch.isfinite(total)
    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def components_names():
    return ("non_negative", "coarse_consistency", "mass_conservation", "gradient", "extreme_structure")


def test_full_diffusion_physics_gradient_flow():
    process = DiffusionProcess(8, "linear")
    model = ConditionalDiffusionUNet(base_channels=4)
    target = torch.rand(1, 1, 16, 16)
    condition = torch.rand_like(target)
    timestep = torch.tensor([3])
    noisy = process.q_sample(log1p_transform(target), timestep)
    predicted_noise = model(noisy, timestep, log1p_transform(condition))
    prediction = inverse_log1p_transform(process.predict_x0(noisy, timestep, predicted_noise).clamp(0.0, 6.0))
    loss, _ = physics_loss(prediction, target, torch.rand(1, 1, 4, 4))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.requires_grad)
