"""Train the Phase 6 conditional precipitation diffusion model."""

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
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionLoss, DiffusionProcess
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.synthetic import generate_paired_synthetic_dataset
from src.downscaling.dataset import split_downscaling_samples


def _load_samples(cfg: Dict[str, Any], seed: int):
    dataset_dir = Path(cfg.get("downscaling", {}).get("paths", {}).get("dataset_dir", "data/downscaling"))
    train_path = dataset_dir / "train" / "samples.pkl"
    val_path = dataset_dir / "validation" / "samples.pkl"
    if train_path.exists() and val_path.exists():
        with open(train_path, "rb") as handle:
            train_samples = pickle.load(handle)
        with open(val_path, "rb") as handle:
            val_samples = pickle.load(handle)
        return train_samples, val_samples
    samples = generate_paired_synthetic_dataset(num_events=8, timesteps_per_event=3, seed=seed)
    train_samples, val_samples, _ = split_downscaling_samples(samples, seed=seed)
    return train_samples, val_samples


def train_diffusion(
    config_path: str = "configs/data.yaml",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    device_override: Optional[str] = None,
    output_checkpoint: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    dcfg = cfg.get("diffusion", {})
    tcfg = dcfg.get("training", {})
    mcfg = dcfg.get("model", {})
    loss_cfg = dcfg.get("loss", {})
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_samples, val_samples = _load_samples(cfg, seed)
    interp = cfg.get("downscaling", {}).get("interpolation", {}).get("default_method", "bilinear")
    train_loader = DataLoader(DiffusionDownscalingDataset(train_samples, interp_method=interp), batch_size=batch_size_override or int(tcfg.get("batch_size", 4)), shuffle=True)
    val_loader = DataLoader(DiffusionDownscalingDataset(val_samples, interp_method=interp), batch_size=batch_size_override or int(tcfg.get("batch_size", 4)), shuffle=False)
    process = DiffusionProcess(int(dcfg.get("timesteps", 100)), dcfg.get("beta_schedule", "cosine")).to(device)
    model = ConditionalDiffusionUNet(**{key: int(mcfg[key]) for key in ("base_channels", "condition_channels", "target_channels") if key in mcfg}).to(device)
    criterion = DiffusionLoss(**{key: loss_cfg[key] for key in ("extreme_enabled", "extreme_percentile", "extreme_weight") if key in loss_cfg})
    optimizer = optim.AdamW(model.parameters(), lr=float(tcfg.get("learning_rate", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    epochs = epochs_override or int(tcfg.get("epochs", 50))
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            target = batch["target"].to(device)
            condition = batch["condition"].to(device)
            t = torch.randint(0, process.timesteps, (target.shape[0],), device=device)
            noise = torch.randn_like(target)
            noisy = process.q_sample(target, t, noise)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy, t, condition), noise, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("gradient_clip", 1.0)))
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(device)
                condition = batch["condition"].to(device)
                t = torch.randint(0, process.timesteps, (target.shape[0],), device=device)
                noise = torch.randn_like(target)
                val_losses.append(float(criterion(model(process.q_sample(target, t, noise), t, condition), noise, target)))
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            checkpoint_path = Path(output_checkpoint or dcfg.get("paths", {}).get("model_checkpoint", "models/diffusion_downscaler_best.pt"))
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "validation_loss": best_val, "config": dcfg, "transform": "log1p"}, checkpoint_path)
    return {"checkpoint_path": str(output_checkpoint or dcfg.get("paths", {}).get("model_checkpoint", "models/diffusion_downscaler_best.pt")), "best_val_loss": best_val, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(train_diffusion(args.config, args.epochs, args.batch_size, args.device, args.checkpoint, args.seed))
