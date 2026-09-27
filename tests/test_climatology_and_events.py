"""
Unit tests for SpatioAI Phase 2 Climatology, Anomaly Detection, and Event Segmentation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.climatology.anomaly import (
    AnomalyCalculator,
    compute_absolute_anomaly,
    compute_standardized_anomaly,
)
from src.climatology.baseline import ClimatologyBaseline
from src.climatology.statistics import compute_group_statistics, create_time_group_key
from src.data.synthetic import (
    generate_synthetic_weather_dataset,
    inject_synthetic_extreme_event,
)
from src.events.detection import ExtremeDetector
from src.events.segmentation import (
    EventSegmenter,
    compute_grid_cell_areas_km2,
)


@pytest.fixture
def synthetic_climatology_dataset() -> xr.Dataset:
    """Fixture generating a 10-day synthetic weather dataset with distinct daily cycles."""
    return generate_synthetic_weather_dataset(
        start_date="2024-01-01",
        end_date="2024-01-10",
        freq="6h",
        lat_min=15.0,
        lat_max=25.0,
        lat_step=1.0,
        lon_min=80.0,
        lon_max=90.0,
        lon_step=1.0,
        seed=101,
    )


def test_climatology_mean(synthetic_climatology_dataset: xr.Dataset):
    """Test computation of climatological mean across time groups."""
    ds = synthetic_climatology_dataset
    group_key = create_time_group_key(ds["time"], method="month_hour")
    stats_ds = compute_group_statistics(
        da=ds["precipitation"],
        group_key=group_key,
        statistics=["mean"],
    )
    assert "precipitation_mean" in stats_ds.data_vars
    assert not np.isnan(stats_ds["precipitation_mean"].values).any()
    assert np.all(stats_ds["precipitation_mean"].values >= 0.0)


def test_climatology_std(synthetic_climatology_dataset: xr.Dataset):
    """Test computation of climatological standard deviation."""
    ds = synthetic_climatology_dataset
    group_key = create_time_group_key(ds["time"], method="month_hour")
    stats_ds = compute_group_statistics(
        da=ds["temperature"],
        group_key=group_key,
        statistics=["std"],
    )
    assert "temperature_std" in stats_ds.data_vars
    assert not np.isnan(stats_ds["temperature_std"].values).any()
    assert np.all(stats_ds["temperature_std"].values >= 0.0)


def test_percentiles(synthetic_climatology_dataset: xr.Dataset):
    """Test percentile calculations (p50, p90, p95, p99)."""
    ds = synthetic_climatology_dataset
    group_key = create_time_group_key(ds["time"], method="month_hour")
    stats_ds = compute_group_statistics(
        da=ds["precipitation"],
        group_key=group_key,
        statistics=["median", "p90", "p95", "p99"],
    )
    assert "precipitation_median" in stats_ds.data_vars
    assert "precipitation_p90" in stats_ds.data_vars
    assert "precipitation_p95" in stats_ds.data_vars
    assert "precipitation_p99" in stats_ds.data_vars

    # Verify monotonic ordering: median <= p90 <= p95 <= p99
    med = stats_ds["precipitation_median"].values
    p90 = stats_ds["precipitation_p90"].values
    p95 = stats_ds["precipitation_p95"].values
    p99 = stats_ds["precipitation_p99"].values

    assert np.all(p90 >= med - 1e-5)
    assert np.all(p95 >= p90 - 1e-5)
    assert np.all(p99 >= p95 - 1e-5)


def test_absolute_anomaly():
    """Test actual - baseline_mean absolute anomaly."""
    actual = xr.DataArray(np.array([10.0, 50.0, 5.0]), dims=["time"])
    baseline_mean = xr.DataArray(np.array([10.0, 20.0, 15.0]), dims=["time"])

    anomaly = compute_absolute_anomaly(actual, baseline_mean)
    expected = np.array([0.0, 30.0, -10.0])
    assert np.allclose(anomaly.values, expected)


def test_standardized_anomaly():
    """Test z-score calculation with zero-std safeguard."""
    actual = xr.DataArray(np.array([25.0, 40.0, 10.0]), dims=["time"])
    mean = xr.DataArray(np.array([20.0, 20.0, 10.0]), dims=["time"])
    std = xr.DataArray(np.array([5.0, 10.0, 0.0]), dims=["time"])  # includes zero std

    z_score = compute_standardized_anomaly(actual, mean, std, eps=1e-3)
    assert np.isclose(z_score.values[0], (25.0 - 20.0) / 5.0)  # +1.0
    assert np.isclose(z_score.values[1], (40.0 - 20.0) / 10.0)  # +2.0
    # Safe handling: not NaN or Inf
    assert not np.isnan(z_score.values[2])
    assert not np.isinf(z_score.values[2])


def test_extreme_threshold(synthetic_climatology_dataset: xr.Dataset):
    """Test percentile and standardized anomaly thresholding."""
    baseline = ClimatologyBaseline(grouping_method="month_hour")
    baseline.fit(synthetic_climatology_dataset)

    detector = ExtremeDetector(baseline=baseline)

    # Test percentile detection
    mask_p95 = detector.detect_by_percentile(
        synthetic_climatology_dataset,
        variable="precipitation",
        percentile_stat="p95",
    )
    assert set(np.unique(mask_p95.values)).issubset({0, 1})
    assert mask_p95.shape == synthetic_climatology_dataset["precipitation"].shape

    # Test standardized anomaly detection
    mask_z2 = detector.detect_by_standardized_anomaly(
        synthetic_climatology_dataset,
        variable="precipitation",
        z_threshold=2.0,
    )
    assert set(np.unique(mask_z2.values)).issubset({0, 1})


def test_connected_components():
    """Test 2D connected component labeling on an artificial extreme storm cluster."""
    lats = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    lons = np.array([80.0, 81.0, 82.0, 83.0, 84.0])

    # Create a 5x5 grid with one 3x3 connected storm cluster
    mask = np.zeros((5, 5), dtype=int)
    mask[1:4, 1:4] = 1

    intensity = np.zeros((5, 5), dtype=float)
    intensity[1:4, 1:4] = 75.0
    intensity[2, 2] = 120.0  # peak intensity

    segmenter = EventSegmenter(connectivity=8, minimum_cells=3, minimum_area_km2=10.0)
    labeled_array, events = segmenter.segment_slice(
        mask_2d=mask,
        intensity_2d=intensity,
        lats=lats,
        lons=lons,
        timestamp="2024-01-01 12:00:00",
    )

    assert len(events) == 1
    ev = events[0]
    assert ev["cell_count"] == 9
    assert ev["max_intensity"] == 120.0
    assert np.isclose(ev["mean_intensity"], (8 * 75.0 + 120.0) / 9.0, atol=0.1)


def test_event_centroid():
    """Test intensity-weighted centroid calculation."""
    lats = np.array([10.0, 11.0, 12.0])
    lons = np.array([80.0, 81.0, 82.0])

    mask = np.zeros((3, 3), dtype=int)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[0, 2] = 1

    intensity = np.zeros((3, 3), dtype=float)
    intensity[0, 0] = 100.0  # heavy rain at lon=80
    intensity[0, 1] = 50.0   # rain at lon=81
    intensity[0, 2] = 100.0  # heavy rain at lon=82

    segmenter = EventSegmenter(connectivity=8, minimum_cells=2, minimum_area_km2=1.0)
    _, events = segmenter.segment_slice(
        mask_2d=mask,
        intensity_2d=intensity,
        lats=lats,
        lons=lons,
        timestamp="2024-01-01 00:00:00",
    )

    assert len(events) == 1
    # Symmetric weights (100, 50, 100) -> centroid is exactly at center lon=81.0
    assert np.isclose(events[0]["centroid_lon"], 81.0, atol=1e-2)
    assert np.isclose(events[0]["centroid_lat"], 10.0, atol=1e-2)


def test_event_area():
    """Test spherical latitude-dependent grid cell area calculation."""
    # 1-degree spacing grids at Equator (0..2 deg), Mid-lat (30..32 deg), High-lat (60..62 deg)
    lats_eq = np.array([0.0, 1.0, 2.0])
    lons_eq = np.array([80.0, 81.0, 82.0])

    areas_eq = compute_grid_cell_areas_km2(lats_eq, lons_eq)
    assert areas_eq.shape == (3, 3)

    # At equator for 1x1 deg cell: Area ~= (111.32 km)^2 ~= 12364 km^2
    assert 12000.0 < areas_eq[0, 0] < 12500.0

    # Test latitude reduction factor: Area(lat) proportional to cos(lat)
    lats_mid = np.array([60.0, 61.0, 62.0])
    areas_mid = compute_grid_cell_areas_km2(lats_mid, lons_eq)
    # cos(60 deg) = 0.5, so area at 60 deg latitude should be approximately half of equatorial area
    assert np.isclose(areas_mid[0, 0], areas_eq[0, 0] * 0.5, rtol=0.05)


def test_event_bbox():
    """Test bounding box (min_lat, max_lat, min_lon, max_lon) extraction."""
    lats = np.array([10.0, 12.0, 14.0, 16.0])
    lons = np.array([70.0, 72.0, 74.0, 76.0])

    mask = np.zeros((4, 4), dtype=int)
    mask[1:3, 1:4] = 1  # lats index 1..2 (12..14), lons index 1..3 (72..76)

    intensity = np.full((4, 4), 60.0)

    segmenter = EventSegmenter(connectivity=8, minimum_cells=2, minimum_area_km2=10.0)
    _, events = segmenter.segment_slice(
        mask_2d=mask,
        intensity_2d=intensity,
        lats=lats,
        lons=lons,
        timestamp="2024-01-01",
    )

    assert len(events) == 1
    ev = events[0]
    assert ev["min_lat"] == 12.0
    assert ev["max_lat"] == 14.0
    assert ev["min_lon"] == 72.0
    assert ev["max_lon"] == 76.0


def test_event_filtering():
    """Test that isolated small pixels below minimum_cells / minimum_area are discarded."""
    lats = np.array([10.0, 11.0, 12.0, 13.0])
    lons = np.array([80.0, 81.0, 82.0, 83.0])

    # 1 isolated pixel
    mask = np.zeros((4, 4), dtype=int)
    mask[0, 0] = 1

    intensity = np.full((4, 4), 100.0)

    # Require at least 4 cells
    segmenter = EventSegmenter(connectivity=8, minimum_cells=4, minimum_area_km2=50.0)
    labeled, events = segmenter.segment_slice(
        mask_2d=mask,
        intensity_2d=intensity,
        lats=lats,
        lons=lons,
        timestamp="2024-01-01",
    )

    assert len(events) == 0
    assert np.all(labeled == 0)


def test_synthetic_extreme_event_detection():
    """
    End-to-end validation test:
    Normal multi-day baseline + Injected localized extreme precipitation storm
    -> Climatology fit -> Extreme anomaly mask -> Connected components -> Extracted event features.
    """
    # 1. Multi-day baseline dataset
    ds_raw = generate_synthetic_weather_dataset(
        start_date="2024-01-01",
        end_date="2024-01-15",
        freq="6h",
        lat_min=10.0,
        lat_max=28.0,
        lat_step=0.5,
        lon_min=70.0,
        lon_max=95.0,
        lon_step=0.5,
        seed=42,
    )

    # 2. Inject localized extreme storm (Peak 150mm at 20.5N, 87.5E)
    target_time = "2024-01-08 12:00:00"
    injected_lat = 20.5
    injected_lon = 87.5
    injected_peak = 150.0

    ds_with_event, meta = inject_synthetic_extreme_event(
        ds=ds_raw,
        target_time=target_time,
        center_lat=injected_lat,
        center_lon=injected_lon,
        radius_deg=1.5,
        peak_intensity=injected_peak,
        variable="precipitation",
    )

    # 3. Fit Climatology baseline with month_hour grouping
    baseline = ClimatologyBaseline(
        grouping_method="month_hour",
        variable_configs={"precipitation": {"statistics": ["mean", "std", "median", "p95"]}},
    )
    baseline.fit(ds_with_event)

    # 4. Detect extreme mask (p95 with 25mm floor)
    detector = ExtremeDetector(baseline=baseline)
    mask_da = detector.detect_by_percentile(
        ds=ds_with_event,
        variable="precipitation",
        percentile_stat="p95",
        min_absolute_val=25.0,
    )

    # 5. Extract connected component events
    segmenter = EventSegmenter(connectivity=8, minimum_cells=4, minimum_area_km2=100.0)
    labeled_da, df_events = segmenter.extract_events(mask_da, ds_with_event["precipitation"])

    # 6. Assertions for validation
    assert len(df_events) >= 1, "At least one candidate extreme event must be detected"

    # Verify event characteristics
    ev = df_events.iloc[0]
    assert ev["cell_count"] >= 4
    assert ev["area_km2"] > 100.0
    assert ev["max_intensity"] >= 120.0  # near 150mm injected peak
    assert ev["mean_intensity"] > 25.0

    # Centroid verification: within 0.5 degrees of injected storm center
    assert np.isclose(ev["centroid_lat"], injected_lat, atol=0.5)
    assert np.isclose(ev["centroid_lon"], injected_lon, atol=0.5)

    # Bounding box contains the storm center
    assert ev["min_lat"] <= injected_lat <= ev["max_lat"]
    assert ev["min_lon"] <= injected_lon <= ev["max_lon"]

