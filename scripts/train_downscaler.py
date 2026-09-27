"""
Training pipeline for U-Net extreme-event weather downscaling in SpatioAI Phase 5.
Supports:
  - Training / Validation data loaders
  - Checkpoint saving & best-model tracking
  - Early stopping & LR scheduling
  - Reproducible random seed
  - Extreme-weighted loss
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.downscaling.cnn import UNetDownscaler
from src.downscaling.dataset import DownscalingDataset
from src.downscaling.losses import get_downscaling_loss
from src.downscaling.synthetic import generate_paired_synthetic_dataset
from src.downscaling.dataset import split_downscaling_samples
from src.utils.logger import get_logger

logger = get_logger("TrainDownscaler")


def train_downscaler(
    config_path: str = "configs/data.yaml",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    output_checkpoint: Optional[str] = None,
    device_override: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("downscaling", {})
    m_cfg = ds_cfg.get("model", {})
    t_cfg = ds_cfg.get("training", {})
    paths_cfg = ds_cfg.get("paths", {})

    epochs = epochs_override or int(t_cfg.get("epochs", 30))
    batch_size = batch_size_override or int(t_cfg.get("batch_size", 4))
    lr = lr_override or float(t_cfg.get("learning_rate", 0.001))
    weight_decay = float(t_cfg.get("weight_decay", 0.0001))
    patience = int(t_cfg.get("patience", 8))
    ckpt_path = Path(output_checkpoint or paths_cfg.get("model_checkpoint", "models/downscaler_unet_best.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Device
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Load dataset splits
    dataset_dir = Path(paths_cfg.get("dataset_dir", "data/downscaling"))
    train_pkl = dataset_dir / "train" / "samples.pkl"
    val_pkl = dataset_dir / "validation" / "samples.pkl"

    if train_pkl.exists() and val_pkl.exists():
        with open(train_pkl, "rb") as f:
            train_samples = pickle.load(f)
        with open(val_pkl, "rb") as f:
            val_samples = pickle.load(f)
    else:
        logger.warning("Pre-generated downscaling data not found. Generating on the fly...")
        all_samples = generate_paired_synthetic_dataset(num_events=20, timesteps_per_event=5, seed=seed)
        train_samples, val_samples, _ = split_downscaling_samples(all_samples, seed=seed)

    train_ds = DownscalingDataset(train_samples, interp_method=ds_cfg.get("interpolation", {}).get("default_method", "bilinear"))
    val_ds = DownscalingDataset(val_samples, interp_method=ds_cfg.get("interpolation", {}).get("default_method", "bilinear"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples.")

    # Initialize U-Net
    model = UNetDownscaler(
        in_channels=int(m_cfg.get("in_channels", 1)),
        out_channels=int(m_cfg.get("out_channels", 1)),
        base_channels=int(m_cfg.get("base_channels", 32)),
        residual_learning=bool(m_cfg.get("residual_learning", True)),
        output_activation=str(m_cfg.get("output_activation", "softplus")),
    ).to(device)

    # Loss and Optimizer
    loss_cfg = t_cfg.get("loss", {})
    criterion = get_downscaling_loss(
        loss_name=loss_cfg.get("name", "weighted_mse"),
        extreme_percentile=float(loss_cfg.get("extreme_percentile", 95.0)),
        extreme_weight=float(loss_cfg.get("extreme_weight", 3.0)),
        gradient_weight=float(loss_cfg.get("gradient_weight", 0.05)),
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info(f"Starting U-Net training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss
        scheduler.step(mean_val_loss)

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_train_loss:.5f} | Val Loss: {mean_val_loss:.5f}")

        # Checkpoint if best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": ds_cfg,
                },
                ckpt_path,
            )
            logger.info(f"  --> Saved new best checkpoint to {ckpt_path} (val_loss={best_val_loss:.5f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(ckpt_path),
        "history": history,
        "epochs_completed": epoch,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Downscaler")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_downscaler(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        output_checkpoint=args.checkpoint,
    )
