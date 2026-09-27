"""Diagnostics for empirical probabilistic outputs."""

from typing import Dict, Optional

import numpy as np


def brier_score(probability, observation) -> float:
    """Compute the binary Brier score ``mean((p-o)^2)``."""
    probability = np.asarray(probability, dtype=float)
    observation = np.asarray(observation, dtype=float)
    if probability.shape != observation.shape:
        raise ValueError("probability and observation must have the same shape")
    return float(np.mean((probability - observation) ** 2))


def reliability_bins(probability, observation, n_bins: int = 10) -> list[Dict[str, float]]:
    """Return reliability-bin means without recalibrating raw probabilities."""
    probability = np.asarray(probability, dtype=float).ravel()
    observation = np.asarray(observation, dtype=float).ravel()
    if probability.shape != observation.shape:
        raise ValueError("probability and observation must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    results = []
    for index in range(n_bins):
        mask = (probability >= edges[index]) & (probability <= edges[index + 1] if index == n_bins - 1 else probability < edges[index + 1])
        results.append({
            "bin_lower": float(edges[index]),
            "bin_upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()) if mask.any() else float("nan"),
            "observed_frequency": float(observation[mask].mean()) if mask.any() else float("nan"),
        })
    return results


def empirical_crps(samples, observation, sample_axis: int = 0) -> float:
    """Compute the empirical CRPS using the energy representation.

    This is an optional diagnostic and requires actual observations/targets.
    """
    ensemble = np.moveaxis(np.asarray(samples, dtype=float), sample_axis, 0)
    observation = np.asarray(observation, dtype=float)
    first = np.mean(np.abs(ensemble - observation), axis=0)
    pairwise = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...]).mean(axis=(0, 1))
    return float(np.mean(first - 0.5 * pairwise))
