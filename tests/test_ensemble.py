"""Tests for Phase 8 conditional ensemble diagnostics."""

import numpy as np
import pytest
import torch

from src.ensemble import (
    brier_score,
    compute_ensemble_statistics,
    empirical_crps,
    reliability_bins,
    threshold_exceedance_probability,
    uncertainty_map,
)


def test_statistics_and_quantiles_preserve_shape():
    samples = torch.arange(20.0).reshape(1, 5, 1, 2, 2)
    statistics = compute_ensemble_statistics(samples)
    assert statistics["mean"].shape == (1, 1, 2, 2)
    assert statistics["median"].shape == statistics["mean"].shape
    assert statistics["q05"].shape == statistics["mean"].shape
    assert torch.all(statistics["std"] >= 0)


def test_uncertainty_constant_and_variable_ensembles():
    constant = torch.ones(1, 5, 1, 2, 2)
    variable = constant.clone()
    variable[:, 1] = 2.0
    assert torch.allclose(uncertainty_map(compute_ensemble_statistics(constant)), torch.zeros(1, 1, 2, 2))
    assert torch.all(uncertainty_map(compute_ensemble_statistics(variable)) > 0)


def test_empirical_exceedance_probability():
    samples = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]], [[[6.0]]], [[[7.0]]], [[[8.0]]], [[[9.0]]], [[[10.0]]]]])
    probability = threshold_exceedance_probability(samples, 5.0)
    assert probability.item() == pytest.approx(0.5)


def test_probabilistic_metrics():
    assert brier_score([0.2, 0.8], [0.0, 1.0]) == pytest.approx(0.04)
    bins = reliability_bins([0.05, 0.95], [0.0, 1.0], n_bins=10)
    assert sum(item["count"] for item in bins) == 2
    assert empirical_crps(np.array([1.0, 2.0, 3.0]), 2.0) == pytest.approx(2.0 / 9.0)
