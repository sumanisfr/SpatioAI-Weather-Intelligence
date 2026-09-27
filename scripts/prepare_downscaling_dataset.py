"""
Data preparation script for Phase 5 Downscaling:
Generates synthetic paired paired (coarse 12 km, high-res ~5 km) weather events,
applies zero-leakage track split, and saves datasets to disk.
"""

import argparse
import pickle
import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import yaml
from src.downscaling.dataset import split_downscaling_samples
from src.downscaling.synthetic import generate_paired_synthetic_dataset
from src.utils.logger import get_logger

logger = get_logger("PrepareDownscalingData")


def prepare_downscaling_dataset(
    config_path: str = "configs/data.yaml",
    num_events: int = 24,
    timesteps_per_event: int = 6,
    seed: int = 42,
) -> Path:
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("downscaling", {})
    input_res_km = float(ds_cfg.get("input_resolution_km", 12.0))
    target_res_km = float(ds_cfg.get("target_resolution_km", 5.0))
    crop_size_deg = float(ds_cfg.get("crop", {}).get("crop_size_deg", 4.0))

    split_cfg = ds_cfg.get("split", {})
    split_method = split_cfg.get("method", "track")
    train_ratio = float(split_cfg.get("train_ratio", 0.70))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))

    out_base = Path(ds_cfg.get("paths", {}).get("dataset_dir", "data/downscaling"))
    train_dir = out_base / "train"
    val_dir = out_base / "validation"
    test_dir = out_base / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating synthetic paired downscaling dataset ({num_events} tracks, {timesteps_per_event} steps, "
        f"low_res={input_res_km} km, high_res={target_res_km} km)..."
    )
    samples = generate_paired_synthetic_dataset(
        num_events=num_events,
        timesteps_per_event=timesteps_per_event,
        crop_size_deg=crop_size_deg,
        high_res_km=target_res_km,
        low_res_km=input_res_km,
        seed=seed,
    )

    train_samples, val_samples, test_samples = split_downscaling_samples(
        samples=samples,
        method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Save splits as pickled dictionaries
    with open(train_dir / "samples.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(val_dir / "samples.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(test_dir / "samples.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    logger.info(f"Dataset preparation complete! Saved to {out_base}:")
    logger.info(f"  - Train: {len(train_samples)} samples in {train_dir}")
    logger.info(f"  - Val:   {len(val_samples)} samples in {val_dir}")
    logger.info(f"  - Test:  {len(test_samples)} samples in {test_dir}")

    return out_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Downscaling Dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--num_events", type=int, default=24)
    parser.add_argument("--timesteps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_downscaling_dataset(
        config_path=args.config,
        num_events=args.num_events,
        timesteps_per_event=args.timesteps,
        seed=args.seed,
    )
