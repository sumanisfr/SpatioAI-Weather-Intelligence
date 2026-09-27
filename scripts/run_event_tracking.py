"""
Run temporal extreme weather event tracking pipeline for SpatioAI Phase 3.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.events.association import EventAssociationConfig
from src.events.tracking import BaselineEventTracker
from src.utils.logger import get_logger

logger = get_logger("RunEventTracking")


def load_config(config_path: str = "configs/data.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_tracking_pipeline(config_path: str = "configs/data.yaml") -> Tuple[Path, Path]:
    """
    Execute temporal event tracking from detected candidate events table.
    """
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    event_cfg = cfg.get("events", {})
    track_cfg = cfg.get("tracking", {})

    target_var = event_cfg.get("variable", "precipitation")
    events_dir = PROJECT_ROOT / data_cfg.get("events_dir", "data/events")

    # Look for candidate events table
    parquet_in = events_dir / f"{target_var}_candidate_events.parquet"
    csv_in = events_dir / f"{target_var}_candidate_events.csv"

    if parquet_in.exists():
        logger.info(f"Loading candidate events from {parquet_in}")
        df_events = pd.read_parquet(str(parquet_in))
    elif csv_in.exists():
        logger.info(f"Loading candidate events from {csv_in}")
        df_events = pd.read_csv(str(csv_in))
    else:
        logger.info(f"Candidate events table not found in {events_dir}. Running anomaly detection first...")
        from scripts.run_anomaly_detection import run_anomaly_pipeline
        saved_events_path = run_anomaly_pipeline(config_path=config_path)
        if str(saved_events_path).endswith(".parquet"):
            df_events = pd.read_parquet(str(saved_events_path))
        else:
            df_events = pd.read_csv(str(saved_events_path))

    if df_events.empty:
        logger.warning("No candidate events available to track.")
        return None, None

    weights = track_cfg.get("weights", {})
    assoc_config = EventAssociationConfig(
        max_centroid_distance_km=float(track_cfg.get("max_centroid_distance_km", 500.0)),
        max_speed_kmh=float(track_cfg.get("max_speed_kmh", 120.0)),
        max_area_change_ratio=float(track_cfg.get("max_area_change_ratio", 5.0)),
        max_missed_steps=int(track_cfg.get("max_missed_steps", 1)),
        cost_threshold=float(track_cfg.get("cost_threshold", 0.85)),
        algorithm=str(track_cfg.get("algorithm", "hungarian")),
        weight_distance=float(weights.get("weight_distance", 0.50)),
        weight_bbox=float(weights.get("weight_bbox", 0.20)),
        weight_area=float(weights.get("weight_area", 0.15)),
        weight_intensity=float(weights.get("weight_intensity", 0.15)),
    )

    tracker = BaselineEventTracker(
        config=assoc_config,
        use_kalman=bool(track_cfg.get("use_kalman", False)),
    )

    df_track_events, df_track_summary = tracker.track(df_events)

    events_out, summary_out = BaselineEventTracker.save_tracking_results(
        df_track_events=df_track_events,
        df_track_summary=df_track_summary,
        output_dir=events_dir,
        prefix=target_var,
        file_format=track_cfg.get("output_format", "parquet"),
    )

    logger.info(f"=== TRACKING RESULTS ===")
    logger.info(f"Formed {len(df_track_summary)} tracks from {len(df_events)} event objects.")
    if not df_track_summary.empty:
        logger.info(f"Track Summary:\n{df_track_summary.to_string(index=False)}")

    return events_out, summary_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal extreme weather event tracking")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config file")
    args = parser.parse_args()

    run_tracking_pipeline(config_path=args.config)
