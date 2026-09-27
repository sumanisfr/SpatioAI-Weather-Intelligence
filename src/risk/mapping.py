"""Spatial risk maps and event-level summaries."""

from typing import Any, Dict, Optional

import numpy as np
import torch

from src.ensemble.probability import threshold_exceedance_probability
from src.ensemble.statistics import compute_ensemble_statistics
from src.ensemble.uncertainty import uncertainty_map
from src.risk.area import affected_area_km2
from src.risk.scoring import transparent_risk_score
from src.risk.severity import severity_index


def build_risk_map(
    samples: torch.Tensor,
    threshold,
    latitudes,
    longitudes,
    probability_threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build empirical probability, uncertainty, severity, and area fields."""
    statistics = compute_ensemble_statistics(samples, sample_dim=1)
    probability = threshold_exceedance_probability(samples, threshold, sample_dim=1)
    severity = severity_index(statistics["mean"], threshold)
    area = affected_area_km2(probability, latitudes, longitudes, probability_threshold)
    score = transparent_risk_score(probability.mean(dim=(-2, -1)), severity.mean(dim=(-2, -1)), area)
    result = {
        "statistics": statistics,
        "exceedance_probability": probability,
        "uncertainty": uncertainty_map(statistics),
        "severity": severity,
        "affected_area_km2": area,
        "affected_area_mask": probability >= probability_threshold,
        "risk_score": score,
        "latitude": np.asarray(latitudes),
        "longitude": np.asarray(longitudes),
    }
    if metadata:
        result["metadata"] = dict(metadata)
    return result


def event_risk_summary(risk_map: Dict[str, Any], threshold: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten a risk map into a machine-readable event summary."""
    probability = risk_map["exceedance_probability"]
    statistics = risk_map["statistics"]
    uncertainty = risk_map["uncertainty"]
    summary = {
        "threshold": float(threshold),
        "max_probability": float(probability.max()),
        "mean_probability": float(probability.mean()),
        "max_predicted_intensity": float(statistics["mean"].max()),
        "affected_area_km2": float(torch.as_tensor(risk_map["affected_area_km2"]).mean()),
        "uncertainty_mean": float(uncertainty.mean()),
        "uncertainty_max": float(uncertainty.max()),
        "risk_score": float(torch.as_tensor(risk_map["risk_score"]).mean()),
    }
    if metadata:
        summary.update(metadata)
    return summary


def temporal_risk_summary(summaries: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Preserve risk summaries in timestamp order without inventing missing leads."""
    return sorted(summaries, key=lambda item: str(item.get("timestamp", "")))
