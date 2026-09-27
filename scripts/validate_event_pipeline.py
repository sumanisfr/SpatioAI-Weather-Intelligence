"""
Validation script for synthetic extreme event detection in SpatioAI Phase 2.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.climatology.anomaly import AnomalyCalculator
from src.climatology.baseline import ClimatologyBaseline
from src.data.synthetic import (
    generate_synthetic_weather_dataset,
    inject_synthetic_extreme_event,
)
from src.events.detection import ExtremeDetector
from src.events.segmentation import EventSegmenter
from src.utils.logger import get_logger

logger = get_logger("ValidateEventPipeline")


def run_validation():
    """
    Run end-to-end validation on a synthetic weather dataset with an injected extreme event.
    """
    logger.info("Generating multi-day synthetic dataset with normal background precipitation...")
    # 15 days of 6-hourly data = 60 timestamps
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

    # Inject extreme precipitation storm over Bay of Bengal / Eastern Coast
    target_storm_time = "2024-01-08 12:00:00"
    injected_lat = 20.5
    injected_lon = 87.5
    injected_peak = 150.0

    logger.info(
        f"Injecting artificial extreme storm at ({injected_lat}N, {injected_lon}E) "
        f"on {target_storm_time} with peak intensity {injected_peak} mm..."
    )

    ds_with_event, meta = inject_synthetic_extreme_event(
        ds=ds_raw,
        target_time=target_storm_time,
        center_lat=injected_lat,
        center_lon=injected_lon,
        radius_deg=1.5,
        peak_intensity=injected_peak,
        variable="precipitation",
    )

    # 1. Fit Climatology Baseline with month_hour grouping (multiple samples per diurnal cycle)
    logger.info("Fitting climatology baseline using 'month_hour' grouping...")
    baseline = ClimatologyBaseline(
        grouping_method="month_hour",
        variable_configs={
            "precipitation": {"statistics": ["mean", "std", "median", "p90", "p95", "p99"]},
            "temperature": {"statistics": ["mean", "std", "p95"]},
        },
    )
    baseline.fit(ds_with_event)

    # 2. Run Anomaly & Extreme Threshold Detection (Percentile and Standardized)
    logger.info("Detecting candidate extreme weather anomaly mask...")
    detector = ExtremeDetector(baseline=baseline)

    # Detect using percentile (p95 with 25mm threshold floor)
    mask_da = detector.detect_by_percentile(
        ds=ds_with_event,
        variable="precipitation",
        percentile_stat="p95",
        min_absolute_val=25.0,
    )

    # 3. Connected Component Segmentation and Event Feature Extraction
    logger.info("Running connected component segmentation and spatial feature extraction...")
    segmenter = EventSegmenter(
        connectivity=8,
        minimum_cells=4,
        minimum_area_km2=100.0,
    )

    labeled_da, df_events = segmenter.extract_events(
        mask_da=mask_da,
        intensity_da=ds_with_event["precipitation"],
    )

    logger.info(f"=== VALIDATION RESULTS ===")
    logger.info(f"Total candidate events detected: {len(df_events)}")

    if not df_events.empty:
        logger.info(f"Detected event table:\n{df_events.to_string(index=False)}")

        for _, ev in df_events.iterrows():
            logger.info(f"--------------------------------------------------")
            logger.info(f"Event ID: {ev['event_id']}")
            logger.info(f"Timestamp: {ev['timestamp']}")
            logger.info(f"Centroid: ({ev['centroid_lat']}N, {ev['centroid_lon']}E) [Expected ~({injected_lat}N, {injected_lon}E)]")
            logger.info(f"Bounding Box: Lat [{ev['min_lat']}, {ev['max_lat']}], Lon [{ev['min_lon']}, {ev['max_lon']}]")
            logger.info(f"Physical Area: {ev['area_km2']} km^2")
            logger.info(f"Cell Count: {ev['cell_count']}")
            logger.info(f"Max Intensity: {ev['max_intensity']} mm [Normal background < 20 mm, Injected peak ~{injected_peak} mm]")
            logger.info(f"Mean Intensity: {ev['mean_intensity']} mm")
            logger.info(f"--------------------------------------------------")

    return df_events, meta


if __name__ == "__main__":
    run_validation()
