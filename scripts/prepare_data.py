"""
Data preparation script for SpatioAI Phase 1.
Loads configuration, generates/ingests data, preprocesses, validates, regrids, and saves to Zarr.
"""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base_loader import WeatherDataLoader
from src.data.era5_loader import ERA5Loader
from src.data.regridding import Regridder
from src.data.synthetic import generate_synthetic_weather_dataset
from src.utils.logger import get_logger

logger = get_logger("PrepareData")


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data(config_path: str = "configs/data.yaml", use_synthetic: bool = True) -> Path:
    """
    Execute full data preparation pipeline.
    """
    cfg = load_config(config_path)
    logger.info(f"Loaded configuration for project: {cfg['project']['name']} (v{cfg['project']['version']})")

    region = cfg.get("region", {})
    variables = cfg.get("variables", [])
    time_cfg = cfg.get("time", {})
    data_cfg = cfg.get("data", {})
    proc_cfg = cfg.get("processing", {})

    zarr_out_dir = PROJECT_ROOT / data_cfg.get("zarr_dir", "data/zarr")
    zarr_out_path = zarr_out_dir / f"{region.get('name', 'region')}_processed.zarr"

    if use_synthetic:
        logger.info("Generating synthetic meteorological dataset for verification...")
        raw_ds = generate_synthetic_weather_dataset(
            start_date=time_cfg.get("start_date", "2024-01-01"),
            end_date=time_cfg.get("end_date", "2024-01-05"),
            freq=time_cfg.get("frequency", "6h"),
            lat_min=region.get("lat_min", 5.0) - 2.0,
            lat_max=region.get("lat_max", 30.0) + 2.0,
            lon_min=region.get("lon_min", 65.0) - 2.0,
            lon_max=region.get("lon_max", 100.0) + 2.0,
        )
    else:
        raw_path = PROJECT_ROOT / data_cfg.get("raw_dir", "data/raw")
        logger.info(f"Looking for raw data in: {raw_path}")
        raw_ds = str(raw_path)

    loader = ERA5Loader(config=cfg)

    # Ingest, preprocess, crop, and validate
    logger.info("Ingesting and processing dataset through pipeline...")
    ds = loader.load(
        source=raw_ds,
        start_date=time_cfg.get("start_date"),
        end_date=time_cfg.get("end_date"),
        region=region,
        variables=variables,
        validate=True,
    )

    logger.info(f"Processed dataset dimensions: {dict(ds.sizes)}")
    logger.info(f"Variables present: {list(ds.data_vars.keys())}")

    # Optional Regridding step (e.g. 12km resolution grid: ~0.1 deg)
    target_res_km = proc_cfg.get("target_resolution_km")
    if target_res_km:
        # Approximate 1 degree ~= 111 km -> 12 km ~= 0.108 deg
        lat_step = target_res_km / 111.0
        regridder = Regridder(
            lat_step=lat_step,
            lat_bounds=(region.get("lat_min", 5.0), region.get("lat_max", 30.0)),
            lon_bounds=(region.get("lon_min", 65.0), region.get("lon_max", 100.0)),
            method=proc_cfg.get("regrid_method", "linear"),
        )
        ds = regridder.regrid(ds)

    # Save to Zarr
    chunks = proc_cfg.get("chunk_size", {"time": 10, "latitude": 50, "longitude": 50})
    if isinstance(chunks, dict):
        ds = ds.chunk(chunks)

    logger.info(f"Saving processed dataset to Zarr at {zarr_out_path}...")
    WeatherDataLoader.save_to_zarr(ds, zarr_out_path, mode="w")

    # Verify reload from Zarr
    reloaded_ds = WeatherDataLoader.load_from_zarr(zarr_out_path)
    logger.info(f"Successfully verified reload from Zarr. Shape: {dict(reloaded_ds.sizes)}")

    return zarr_out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare meteorological dataset for SpatioAI")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset")
    args = parser.parse_args()

    prepare_data(config_path=args.config, use_synthetic=args.synthetic)
