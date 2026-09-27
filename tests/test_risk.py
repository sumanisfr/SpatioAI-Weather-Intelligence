"""Tests for Phase 8 deterministic risk calculations."""

import pytest
import torch

from src.risk import (
    affected_area_km2,
    build_risk_map,
    cell_area_km2,
    event_risk_summary,
    resolve_thresholds,
    severity_index,
    temporal_risk_summary,
    transparent_risk_score,
)


def test_geographic_cell_area_and_affected_area():
    latitudes = [0.0, 60.0]
    longitudes = [70.0, 71.0]
    areas = cell_area_km2(latitudes, longitudes)
    assert areas.shape == (2, 2)
    assert areas[0, 0] > areas[1, 0]
    probability = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert affected_area_km2(probability, latitudes, longitudes, 0.5).item() == pytest.approx(float(areas[0, 0] + areas[1, 1]))


def test_threshold_modes_and_severity_score():
    assert resolve_thresholds("fixed", [25, 50]) == [25.0, 50.0]
    with pytest.raises(ValueError):
        resolve_thresholds("climatological_percentile", percentiles=[95])
    assert severity_index(torch.tensor([100.0]), 50.0).item() == pytest.approx(2.0)
    assert transparent_risk_score(torch.tensor([0.5]), torch.tensor([2.0]), torch.tensor([1000.0])).item() == pytest.approx(1.0)


def test_risk_map_and_event_metadata():
    samples = torch.tensor([[[[[10.0, 60.0], [20.0, 80.0]]], [[[20.0, 40.0], [30.0, 100.0]]], [[[30.0, 70.0], [10.0, 120.0]]]]])
    metadata = {"event_id": "EV1", "track_id": "TRK1", "timestamp": "2024-01-01T00:00:00"}
    risk_map = build_risk_map(samples, 50.0, [0.0, 60.0], [70.0, 71.0], metadata=metadata)
    summary = event_risk_summary(risk_map, 50.0, metadata)
    assert risk_map["exceedance_probability"].shape == (1, 1, 2, 2)
    assert 0.0 <= summary["max_probability"] <= 1.0
    assert summary["event_id"] == "EV1"
    assert summary["track_id"] == "TRK1"
    assert temporal_risk_summary([{"timestamp": "b"}, {"timestamp": "a"}])[0]["timestamp"] == "a"
