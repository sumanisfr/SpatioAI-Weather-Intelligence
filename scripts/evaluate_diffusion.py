"""Evaluate diffusion samples against Phase 5 interpolation and U-Net baselines."""

import argparse
import json
import pickle
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import yaml

from src.downscaling.cnn import UNetDownscaler
from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess, log1p_transform
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.diffusion_sampler import generate
from src.downscaling.evaluation import compare_downscaling_methods
from src.downscaling.interpolation import interpolate_field
from src.downscaling.synthetic import generate_paired_synthetic_dataset


def evaluate_diffusion(config_path="configs/data.yaml", checkpoint_path=None, unet_checkpoint=None, seed=42):
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    dcfg = cfg.get("diffusion", {})
    paths = cfg.get("downscaling", {}).get("paths", {})
    test_path = Path(paths.get("dataset_dir", "data/downscaling")) / "test" / "samples.pkl"
    if test_path.exists():
        with open(test_path, "rb") as handle:
            samples = pickle.load(handle)
    else:
        samples = generate_paired_synthetic_dataset(num_events=3, timesteps_per_event=1, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionUNet(**{key: int(dcfg.get("model", {}).get(key, default)) for key, default in (("base_channels", 32), ("condition_channels", 1), ("target_channels", 1))}).to(device)
    checkpoint = Path(checkpoint_path or dcfg.get("paths", {}).get("model_checkpoint", "models/diffusion_downscaler_best.pt"))
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    model.eval()
    unet = UNetDownscaler(in_channels=1, out_channels=1, base_channels=32, residual_learning=True, output_activation="softplus").to(device)
    unet_path = Path(unet_checkpoint or paths.get("model_checkpoint", "models/downscaler_unet_best.pt"))
    if unet_path.exists():
        unet_checkpoint_data = torch.load(unet_path, map_location=device)
        unet.load_state_dict(unet_checkpoint_data.get("model_state_dict", unet_checkpoint_data))
    unet.eval()
    process = DiffusionProcess(int(dcfg.get("timesteps", 100)), dcfg.get("beta_schedule", "cosine")).to(device)
    predictions = {"Nearest": [], "Bilinear": [], "Bicubic": [], "U-Net": [], "Diffusion": []}
    targets = []
    dataset = DiffusionDownscalingDataset(samples)
    for index, item in enumerate(samples):
        nearest = interpolate_field(item["low_res"], item["low_lats"], item["low_lons"], item["high_lats"], item["high_lons"], method="nearest")
        bilinear = interpolate_field(item["low_res"], item["low_lats"], item["low_lons"], item["high_lats"], item["high_lons"], method="bilinear")
        bicubic = interpolate_field(item["low_res"], item["low_lats"], item["low_lons"], item["high_lats"], item["high_lons"], method="bicubic")
        batch = dataset[index]
        with torch.no_grad():
            unet_prediction = unet(batch["baseline"].unsqueeze(0).to(device)).squeeze().cpu().numpy()
        samples_out = generate(model, process, batch["condition"].unsqueeze(0).to(device), num_samples=int(dcfg.get("sampling", {}).get("num_samples", 5)), seed=seed + index)
        predictions["Nearest"].append(nearest)
        predictions["Bilinear"].append(bilinear)
        predictions["Bicubic"].append(bicubic)
        predictions["U-Net"].append(unet_prediction)
        predictions["Diffusion"].append(samples_out.mean(dim=1).squeeze().cpu().numpy())
        targets.append(item["high_res"])
    target_array = np.stack(targets)
    result = compare_downscaling_methods({key: np.stack(value) for key, value in predictions.items()}, target_array)
    output_dir = Path(dcfg.get("paths", {}).get("output_dir", "data/outputs/downscaling"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "diffusion_metrics.json", "w") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame(result).T.to_csv(output_dir / "model_comparison.csv")
    return pd.DataFrame(result).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    print(evaluate_diffusion(args.config, args.checkpoint).to_string())
