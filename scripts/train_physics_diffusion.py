"""Train controlled standard, extreme-aware, or physics-informed diffusion experiments."""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.train_diffusion import _load_samples
from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionLoss, DiffusionProcess, inverse_log1p_transform
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.physics.losses import physics_loss


def train_physics_diffusion(
    config_path: str = "configs/data.yaml",
    experiment: str = "physics",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    checkpoint: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one controlled experiment and return measured histories."""
    if experiment not in {"standard", "extreme", "physics"}:
        raise ValueError("experiment must be standard, extreme, or physics")
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    dcfg = cfg.get("diffusion", {})
    pcfg = cfg.get("physics", {})
    tcfg = dcfg.get("training", {})
    mcfg = dcfg.get("model", {})
    train_samples, val_samples = _load_samples(cfg, seed)
    # Geographic coarse grids can have one-cell longitude differences across
    # samples; physics losses therefore process one sample at a time.
    loader_kwargs = {"batch_size": 1}
    train_loader = DataLoader(DiffusionDownscalingDataset(train_samples), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(DiffusionDownscalingDataset(val_samples), shuffle=False, **loader_kwargs)
    compute_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process = DiffusionProcess(int(dcfg.get("timesteps", 100)), dcfg.get("beta_schedule", "cosine")).to(compute_device)
    model = ConditionalDiffusionUNet(**{key: int(mcfg[key]) for key in ("base_channels", "condition_channels", "target_channels") if key in mcfg}).to(compute_device)
    data_loss = DiffusionLoss(
        extreme_enabled=experiment in {"extreme", "physics"},
        extreme_percentile=float(dcfg.get("loss", {}).get("extreme_percentile", 95.0)),
        extreme_weight=float(dcfg.get("loss", {}).get("extreme_weight", 2.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("learning_rate", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    physics_config = {"weights": pcfg.get("weights", {}), "extreme_percentile": pcfg.get("extreme_percentile", 95.0)}
    n_epochs = epochs or int(tcfg.get("epochs", 50))
    history = {"data_loss": [], "physics_loss": [], "total_loss": [], "components": []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        totals = {"data": [], "physics": [], "total": [], "components": {}}
        for batch in train_loader:
            target = batch["target"].to(compute_device)
            condition = batch["condition"].to(compute_device)
            t = torch.randint(0, process.timesteps, (target.shape[0],), device=compute_device)
            noise = torch.randn_like(target)
            noisy = process.q_sample(target, t, noise)
            predicted_noise = model(noisy, t, condition)
            diffusion_value = data_loss(predicted_noise, noise, target)
            total = diffusion_value
            components = {}
            if experiment == "physics":
                x0_transformed = process.predict_x0(noisy, t, predicted_noise)
                # Physics regularization uses a bounded physical proxy while
                # the diffusion objective still trains the unconstrained x0 estimate.
                x0_transformed = torch.nan_to_num(x0_transformed, nan=0.0, posinf=6.0, neginf=0.0).clamp(0.0, 6.0)
                physical_prediction = inverse_log1p_transform(x0_transformed)
                physical_target = batch["target_physical"].to(compute_device)
                coarse = batch["coarse_physical"].to(compute_device)
                coordinates = {"high_lats": batch["high_lats"][0].to(compute_device), "coarse_lats": batch["coarse_lats"][0].to(compute_device)}
                physical_value, components = physics_loss(physical_prediction, physical_target, coarse, coordinates, physics_config)
                total = total + float(pcfg.get("weight", 1.0)) * physical_value
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("gradient_clip", 1.0)))
            optimizer.step()
            totals["data"].append(float(diffusion_value.detach()))
            totals["physics"].append(float((total - diffusion_value).detach()))
            totals["total"].append(float(total.detach()))
            for name, value in components.items():
                totals["components"].setdefault(name, []).append(float(value.detach()))
        history["data_loss"].append(float(np.mean(totals["data"])))
        history["physics_loss"].append(float(np.mean(totals["physics"])))
        history["total_loss"].append(float(np.mean(totals["total"])))
        history["components"].append({name: float(np.mean(values)) for name, values in totals["components"].items()})
        print(f"epoch={epoch} experiment={experiment} data={history['data_loss'][-1]:.6f} physics={history['physics_loss'][-1]:.6f} total={history['total_loss'][-1]:.6f}")
    output = Path(checkpoint or f"models/diffusion_{experiment}_physics.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": dcfg, "physics_config": pcfg, "experiment": experiment, "history": history}, output)
    return {"experiment": experiment, "checkpoint_path": str(output), "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--experiment", choices=("standard", "extreme", "physics"), default="physics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(train_physics_diffusion(args.config, args.experiment, args.epochs, args.batch_size, args.device, args.checkpoint, args.seed))
