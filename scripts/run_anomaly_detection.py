"""
Run end-to-end extreme anomaly detection and candidate event extraction pipeline.
"""

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.climatology.baseline import ClimatologyBaseline
from src.data.base_loader import WeatherDataLoader
from src.data.synthetic import generate_synthetic_weather_dataset
from src.events.detection import ExtremeDetector
from src.events.segmentation import EventSegmenter
from src.utils.logger import get_logger

logger = get_logger("RunAnomalyDetection")


def load_config(config_path: str = "configs/data.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_anomaly_pipeline(config_path: str = "configs/data.yaml") -> Path:
    """
    Execute Anomaly Detection -> Binary Mask -> Connected Components -> Event Table.
    """
    cfg = load_config(config_path)
    region_cfg = cfg.get("region", {})
    data_cfg = cfg.get("data", {})
    event_cfg = cfg.get("events", {})

    target_var = event_cfg.get("variable", "precipitation")

    clim_path = PROJECT_ROOT / data_cfg.get("climatology_dir", "data/climatology") / f"{region_cfg.get('name', 'region')}_climatology.zarr"
    zarr_in_path = PROJECT_ROOT / data_cfg.get("zarr_dir", "data/zarr") / f"{region_cfg.get('name', 'region')}_processed.zarr"
    events_out_dir = PROJECT_ROOT / data_cfg.get("events_dir", "data/events")
    events_out_path = events_out_dir / f"{target_var}_candidate_events.{event_cfg.get('output_format', 'parquet')}"

    # Load baseline
    baseline = ClimatologyBaseline()
    if not clim_path.exists():
        logger.info(f"Climatology baseline not found at {clim_path}. Building baseline first...")
        from scripts.build_climatology import build_climatology
        build_climatology(config_path=config_path)

    baseline.load(clim_path)

    # Load weather data
    if zarr_in_path.exists():
        weather_ds = WeatherDataLoader.load_from_zarr(zarr_in_path)
    else:
        logger.info("Generating synthetic weather dataset for anomaly detection...")
        weather_ds = generate_synthetic_weather_dataset(
            start_date="2024-01-01",
            end_date="2024-01-05",
            freq="6h",
            lat_min=region_cfg.get("lat_min", 5.0),
            lat_max=region_cfg.get("lat_max", 30.0),
            lon_min=region_cfg.get("lon_min", 65.0),
            lon_max=region_cfg.get("lon_max", 100.0),
        )

    # 1. Detect candidate extreme mask
    detector = ExtremeDetector(baseline=baseline, config=event_cfg)
    mask_da = detector.detect(weather_ds, variable=target_var)

    # 2. Segment connected components and extract event features
    segmenter = EventSegmenter(
        connectivity=event_cfg.get("connectivity", 8),
        minimum_cells=event_cfg.get("minimum_cells", 4),
        minimum_area_km2=event_cfg.get("minimum_area_km2", 50.0),
    )

    labeled_da, df_events = segmenter.extract_events(
        mask_da=mask_da,
        intensity_da=weather_ds[target_var],
    )

    # 3. Save candidate events table
    saved_path = EventSegmenter.save_events_table(
        df_events=df_events,
        output_path=events_out_path,
        file_format=event_cfg.get("output_format", "parquet"),
    )

    if not df_events.empty:
        logger.info(f"Top detected candidate events summary:\n{df_events.head(5).to_string()}")
    else:
        logger.info("No candidate events passed the area/cell filters.")

    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect extreme anomalies and extract candidate events")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config file")
    args = parser.parse_args()

    run_anomaly_pipeline(config_path=args.config)
