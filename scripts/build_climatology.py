"""
Build historical climatological baseline dataset for SpatioAI Phase 2.
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
from src.utils.logger import get_logger

logger = get_logger("BuildClimatology")


def load_config(config_path: str = "configs/data.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_climatology(config_path: str = "configs/data.yaml") -> Path:
    """
    Build and save climatology baseline dataset to data/climatology/.
    """
    cfg = load_config(config_path)
    logger.info("Initializing Climatology Builder...")

    clim_cfg = cfg.get("climatology", {})
    region_cfg = cfg.get("region", {})
    data_cfg = cfg.get("data", {})

    grouping_method = clim_cfg.get("grouping", {}).get("method", "dayofyear_hour")
    var_configs = clim_cfg.get("variables", {
        "precipitation": {"statistics": ["mean", "std", "median", "p90", "p95", "p99"]},
        "temperature": {"statistics": ["mean", "std", "p90", "p95"]},
    })

    # Source Zarr path
    zarr_in_path = PROJECT_ROOT / data_cfg.get("zarr_dir", "data/zarr") / f"{region_cfg.get('name', 'region')}_processed.zarr"
    clim_out_dir = PROJECT_ROOT / data_cfg.get("climatology_dir", "data/climatology")
    clim_out_path = clim_out_dir / f"{region_cfg.get('name', 'region')}_climatology.zarr"

    if zarr_in_path.exists():
        logger.info(f"Opening processed weather data from Zarr: {zarr_in_path}")
        ds = WeatherDataLoader.load_from_zarr(zarr_in_path)
    else:
        logger.warning(f"Zarr store {zarr_in_path} not found. Generating synthetic dataset for climatology build...")
        ds = generate_synthetic_weather_dataset(
            start_date="2024-01-01",
            end_date="2024-01-15",
            freq="6h",
            lat_min=region_cfg.get("lat_min", 5.0),
            lat_max=region_cfg.get("lat_max", 30.0),
            lon_min=region_cfg.get("lon_min", 65.0),
            lon_max=region_cfg.get("lon_max", 100.0),
        )

    baseline = ClimatologyBaseline(
        grouping_method=grouping_method,
        variable_configs=var_configs,
    )

    baseline.fit(ds)
    baseline.save(clim_out_path, mode="w")

    # Verify reload
    reloaded_baseline = ClimatologyBaseline()
    reloaded_ds = reloaded_baseline.load(clim_out_path)
    logger.info(f"Verified climatology baseline reload. Variables: {list(reloaded_ds.data_vars.keys())}")

    return clim_out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build climatological baseline dataset")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configuration file")
    args = parser.parse_args()

    build_climatology(config_path=args.config)
