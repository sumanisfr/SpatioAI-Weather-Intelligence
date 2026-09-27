"""Run standard/extreme/physics synthetic ablations and save measured artifacts."""

import argparse
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from scripts.train_physics_diffusion import train_physics_diffusion
from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess, inverse_log1p_transform
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.evaluation import compute_extreme_downscaling_metrics
from src.physics.losses import physics_loss
from src.downscaling.synthetic import generate_paired_synthetic_dataset


def evaluate_physics_diffusion(config_path: str = "configs/data.yaml", epochs: int = 1, device: str = "cpu", seed: int = 42) -> pd.DataFrame:
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    dcfg = cfg.get("diffusion", {})
    pcfg = cfg.get("physics", {})
    output_dir = Path(dcfg.get("paths", {}).get("output_dir", "data/outputs/downscaling"))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    experiment_results = {}
    for experiment in ("standard", "extreme", "physics"):
        result = train_physics_diffusion(config_path, experiment, epochs=epochs, device=device, checkpoint=str(output_dir / f"diffusion_{experiment}_ablation.pt"), seed=seed)
        experiment_results[experiment] = result
        history = result["history"]
        rows.append({"model": experiment, "data_loss": history["data_loss"][-1], "physics_loss": history["physics_loss"][-1], "total_loss": history["total_loss"][-1]})
    samples = generate_paired_synthetic_dataset(num_events=1, timesteps_per_event=1, seed=seed)
    item = DiffusionDownscalingDataset(samples)[0]
    target = item["target_physical"].unsqueeze(0)
    coarse = item["coarse_physical"].unsqueeze(0)
    coordinates = {"high_lats": item["high_lats"], "coarse_lats": item["coarse_lats"]}
    process = DiffusionProcess(int(dcfg.get("timesteps", 100)), dcfg.get("beta_schedule", "cosine"))
    for row in rows:
        experiment = row["model"]
        model = ConditionalDiffusionUNet(**{key: int(dcfg.get("model", {}).get(key, default)) for key, default in (("base_channels", 32), ("condition_channels", 1), ("target_channels", 1))})
        state = torch.load(experiment_results[experiment]["checkpoint_path"], map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.no_grad():
            t = torch.zeros(1, dtype=torch.long)
            predicted_noise = model(item["target"].unsqueeze(0), t, item["condition"].unsqueeze(0))
            prediction = inverse_log1p_transform(process.predict_x0(item["target"].unsqueeze(0), t, predicted_noise).clamp(0.0, 6.0))
        metrics = compute_extreme_downscaling_metrics(prediction.squeeze().numpy(), target.squeeze().numpy())
        _, components = physics_loss(prediction, target, coarse, coordinates, {"weights": pcfg.get("weights", {}), "extreme_percentile": pcfg.get("extreme_percentile", 95.0)})
        row.update({key: value for key, value in metrics.items() if key in ("mae", "rmse", "correlation", "ssim", "bias_p95", "bias_p99", "peak_error", "area_error", "gradient_error")})
        row.update({f"physics_{key}": float(value) for key, value in components.items()})
    dataframe = pd.DataFrame(rows).set_index("model")
    dataframe.to_csv(output_dir / "physics_comparison.csv")
    with open(output_dir / "physics_metrics.json", "w") as handle:
        json.dump(dataframe.reset_index().to_dict(orient="records"), handle, indent=2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, field) in zip(axes, (("Target", target), ("Coarse", coarse), ("Physics estimate", prediction))):
        axis.imshow(field.squeeze().numpy(), cmap="Blues")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "physics_downscaling_comparison.png", dpi=150)
    plt.close(fig)
    return dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(evaluate_physics_diffusion(args.config, args.epochs, args.device, args.seed).to_string())
