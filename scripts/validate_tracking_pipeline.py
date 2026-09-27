"""
Synthetic end-to-end validation script for SpatioAI Phase 3 Temporal Event Tracking.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.events.association import EventAssociationConfig
from src.events.tracking import BaselineEventTracker
from src.utils.logger import get_logger

logger = get_logger("ValidateTrackingPipeline")


def create_synthetic_event_sequences() -> pd.DataFrame:
    """
    Generate synthetic deterministic multi-track event sequences.
    System 1: Northwest propagating Bay of Bengal cyclonic storm (6 timestamps).
    System 2: Northeast propagating Arabian Sea convective system (6 timestamps).
    """
    timestamps = [
        "2024-01-01 00:00:00",
        "2024-01-01 06:00:00",
        "2024-01-01 12:00:00",
        "2024-01-01 18:00:00",
        "2024-01-02 00:00:00",
        "2024-01-02 06:00:00",
    ]

    events: List[Dict[str, Any]] = []

    # System 1: Bay of Bengal track (NW propagation)
    sys1_coords = [
        (18.0, 89.0),
        (18.6, 88.1),
        (19.3, 87.2),
        (20.0, 86.3),
        (20.7, 85.4),
        (21.4, 84.5),
    ]

    for step, (ts, (lat, lon)) in enumerate(zip(timestamps, sys1_coords)):
        events.append({
            "event_id": f"EV_SYS1_T{step:02d}",
            "timestamp": ts,
            "cell_count": 85 + step * 5,
            "area_km2": 25000.0 + step * 1200.0,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "min_lat": lat - 1.5,
            "max_lat": lat + 1.5,
            "min_lon": lon - 1.5,
            "max_lon": lon + 1.5,
            "max_intensity": 140.0 + step * 3.0,
            "mean_intensity": 65.0,
        })

    # System 2: Arabian Sea track (NE propagation)
    sys2_coords = [
        (9.0, 74.0),
        (9.4, 74.3),
        (9.8, 74.6),
        (10.2, 74.9),
        (10.6, 75.2),
        (11.0, 75.5),
    ]

    for step, (ts, (lat, lon)) in enumerate(zip(timestamps, sys2_coords)):
        events.append({
            "event_id": f"EV_SYS2_T{step:02d}",
            "timestamp": ts,
            "cell_count": 50 + step * 2,
            "area_km2": 15000.0 + step * 500.0,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "min_lat": lat - 1.0,
            "max_lat": lat + 1.0,
            "min_lon": lon - 1.0,
            "max_lon": lon + 1.0,
            "max_intensity": 95.0 + step * 2.0,
            "mean_intensity": 45.0,
        })

    return pd.DataFrame(events)


def run_validation():
    """Execute synthetic tracking validation."""
    logger.info("Generating synthetic multi-event tracking sequences...")
    df_synthetic_events = create_synthetic_event_sequences()
    logger.info(f"Input events generated: {len(df_synthetic_events)} across 6 timestamps.")

    config = EventAssociationConfig(
        max_centroid_distance_km=500.0,
        max_speed_kmh=120.0,
        max_area_change_ratio=3.0,
        max_missed_steps=1,
        algorithm="hungarian",
    )

    tracker = BaselineEventTracker(config=config, use_kalman=False)
    df_track_events, df_track_summary = tracker.track(df_synthetic_events)

    logger.info("==================================================")
    logger.info("SYNTHETIC TRACKING VALIDATION RESULTS")
    logger.info("==================================================")
    logger.info(f"Total Input Events: {len(df_synthetic_events)}")
    logger.info(f"Total Tracks Created: {len(df_track_summary)}")

    assert len(df_track_summary) == 2, f"Expected 2 tracks, got {len(df_track_summary)}"
    assert len(df_track_events) == 12, f"Expected 12 tracked event steps, got {len(df_track_events)}"

    for _, trk in df_track_summary.iterrows():
        logger.info("--------------------------------------------------")
        logger.info(f"Track ID: {trk['track_id']}")
        logger.info(f"Events in Track: {trk['num_events']}")
        logger.info(f"Duration: {trk['duration_hours']} hours ({trk['start_time']} to {trk['end_time']})")
        logger.info(f"Start Location: ({trk['start_lat']}N, {trk['start_lon']}E)")
        logger.info(f"End Location: ({trk['end_lat']}N, {trk['end_lon']}E)")
        logger.info(f"Total Displacement: {trk['total_displacement_km']} km")
        logger.info(f"Cumulative Distance: {trk['cumulative_distance_km']} km")
        logger.info(f"Mean Speed: {trk['mean_speed_kmh']} km/h (Max: {trk['max_speed_kmh']} km/h)")
        logger.info(f"Net Bearing: {trk['net_bearing_deg']}°")
        logger.info(f"Max Intensity: {trk['max_intensity']} mm")
        logger.info(f"Max Area: {trk['max_area_km2']} km^2")

    logger.info("--------------------------------------------------")
    logger.info("Validation: PASSED")

    # Save to data/events/
    out_dir = PROJECT_ROOT / "data" / "events"
    BaselineEventTracker.save_tracking_results(
        df_track_events=df_track_events,
        df_track_summary=df_track_summary,
        output_dir=out_dir,
        prefix="synthetic_validation",
        file_format="parquet",
    )

    return df_track_events, df_track_summary


if __name__ == "__main__":
    run_validation()
