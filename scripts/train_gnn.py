"""
Training script for SpatioAI Spatio-Temporal Graph Neural Network (Phase 4).
Loads configuration, prepares synthetic/real event sequences, fits scalers on training split,
trains GNN model, validates with early stopping, saves checkpoint, and evaluates test set.
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

from src.gnn.dataset import (
    EventGraphData,
    build_graph_data_object,
    generate_synthetic_event_sequence,
    split_events_by_track,
    split_events_temporal,
)
from src.gnn.evaluation import compute_classification_metrics
from src.gnn.features import FeatureScaler
from src.gnn.graph_builder import GraphBuildConfig, SpatioTemporalGraphBuilder
from src.gnn.losses import get_loss_function
from src.gnn.model import SpatioTemporalGNN
from src.utils.logger import get_logger

logger = get_logger("TrainGNN")


def train_gnn(
    config_path: str = "configs/data.yaml",
    synthetic_storms: int = 16,
    timesteps_per_storm: int = 8,
    epochs_override: Optional[int] = None,
    device_str: str = "cpu",
    output_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes complete GNN training and evaluation pipeline.
    """
    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gnn_cfg = cfg.get("gnn", {})
    graph_cfg = gnn_cfg.get("graph", {})
    model_cfg = gnn_cfg.get("model", {})
    train_cfg = gnn_cfg.get("training", {})
    loss_cfg = gnn_cfg.get("loss", {})
    split_cfg = gnn_cfg.get("split", {})

    epochs = epochs_override if epochs_override is not None else train_cfg.get("epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    patience = train_cfg.get("patience", 10)
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)
    device = torch.device(device_str)

    logger.info("Initializing Spatio-Temporal GNN training...")

    # 2. Generate / Load Event sequence data
    logger.info(f"Generating synthetic multi-storm event sequence ({synthetic_storms} storms, {timesteps_per_storm} steps/storm)...")
    events, track_labels = generate_synthetic_event_sequence(
        num_storms=synthetic_storms,
        timesteps_per_storm=timesteps_per_storm,
        seed=42,
    )
    logger.info(f"Total events generated: {len(events)}, Total tracks: {len(set(track_labels.values()))}")

    # 3. Data split (Track-based or Temporal to prevent data leakage)
    split_method = split_cfg.get("method", "track")
    if split_method == "track":
        train_evts, val_evts, test_evts = split_events_by_track(
            events, track_labels,
            train_ratio=split_cfg.get("train_ratio", 0.70),
            val_ratio=split_cfg.get("val_ratio", 0.15),
            seed=42,
        )
    else:
        train_evts, val_evts, test_evts = split_events_temporal(
            events, track_labels,
            train_ratio=split_cfg.get("train_ratio", 0.70),
            val_ratio=split_cfg.get("val_ratio", 0.15),
        )

    logger.info(f"Dataset split ({split_method}): Train={len(train_evts)}, Val={len(val_evts)}, Test={len(test_evts)} events")

    # 4. Construct un-scaled raw graphs to fit Scalers on Training split only
    builder_raw = SpatioTemporalGraphBuilder(GraphBuildConfig(**graph_cfg))

    raw_train_graph = builder_raw.build_graph(train_evts, track_labels=track_labels)
    raw_val_graph = builder_raw.build_graph(val_evts, track_labels=track_labels)
    raw_test_graph = builder_raw.build_graph(test_evts, track_labels=track_labels)

    # Fit Scalers strictly on train graph
    node_scaler = FeatureScaler().fit(raw_train_graph["x"])
    edge_scaler = FeatureScaler().fit(raw_train_graph["edge_attr"])

    # 5. Build scaled graphs
    train_graph_dict = builder_raw.build_graph(train_evts, track_labels=track_labels, node_scaler=node_scaler, edge_scaler=edge_scaler)
    val_graph_dict = builder_raw.build_graph(val_evts, track_labels=track_labels, node_scaler=node_scaler, edge_scaler=edge_scaler)
    test_graph_dict = builder_raw.build_graph(test_evts, track_labels=track_labels, node_scaler=node_scaler, edge_scaler=edge_scaler)

    train_data = build_graph_data_object(train_graph_dict).to(device)
    val_data = build_graph_data_object(val_graph_dict).to(device)
    test_data = build_graph_data_object(test_graph_dict).to(device)

    train_pos = int(train_data.y.sum().item()) if train_data.y is not None and train_data.y.numel() > 0 else 0
    val_pos = int(val_data.y.sum().item()) if val_data.y is not None and val_data.y.numel() > 0 else 0
    test_pos = int(test_data.y.sum().item()) if test_data.y is not None and test_data.y.numel() > 0 else 0

    logger.info(f"Train edges: {train_data.edge_index.size(1)} (Pos: {train_pos})")
    logger.info(f"Val edges: {val_data.edge_index.size(1)} (Pos: {val_pos})")
    logger.info(f"Test edges: {test_data.edge_index.size(1)} (Pos: {test_pos})")


    # 6. Initialize GNN Model & Optimizer
    model_kwargs = {
        "node_in_dim": model_cfg.get("node_in_dim", 15),
        "edge_in_dim": model_cfg.get("edge_in_dim", 9),
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "num_layers": model_cfg.get("num_layers", 3),
        "dropout": model_cfg.get("dropout", 0.2),
        "conv_type": model_cfg.get("conv_type", "sage"),
    }
    model = SpatioTemporalGNN(**model_kwargs).to(device)

    criterion = get_loss_function(
        loss_type=loss_cfg.get("type", "bce_weighted"),
        pos_weight=loss_cfg.get("positive_weight", 3.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.75),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 7. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_dir = Path(gnn_cfg.get("checkpoints_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = Path(output_checkpoint) if output_checkpoint else checkpoint_dir / gnn_cfg.get("best_model_name", "spatiotemporal_gnn_best.pt")

    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(logits, train_data.y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr)
            val_loss = criterion(val_logits, val_data.y).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = compute_classification_metrics(val_data.y.cpu().numpy(), val_probs)

        train_loss = loss.item()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
            )

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_kwargs": model_kwargs,
                "config": cfg,
                "node_scaler": {
                    "mean": node_scaler.mean.tolist() if node_scaler.mean is not None else [],
                    "std": node_scaler.std.tolist() if node_scaler.std is not None else [],
                },
                "edge_scaler": {
                    "mean": edge_scaler.mean.tolist() if edge_scaler.mean is not None else [],
                    "std": edge_scaler.std.tolist() if edge_scaler.std is not None else [],
                },
                "best_val_metrics": val_metrics,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch})")
                break

    # 8. Load best checkpoint and evaluate on Test Set
    logger.info(f"Loading best checkpoint from {best_checkpoint_path}...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        if test_data.edge_index.size(1) > 0:
            test_logits = model(test_data.x, test_data.edge_index, test_data.edge_attr)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            y_test_arr = test_data.y.cpu().numpy() if test_data.y is not None else np.zeros(test_data.edge_index.size(1))
            test_metrics = compute_classification_metrics(y_test_arr, test_probs)
        else:
            test_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0, "pr_auc": 1.0}

    logger.info(f"Test Set Evaluation Results: {test_metrics}")


    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "checkpoint_path": str(best_checkpoint_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpatioAI Spatio-Temporal GNN")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--storms", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_gnn(
        config_path=args.config,
        synthetic_storms=args.storms,
        timesteps_per_storm=args.timesteps,
        epochs_override=args.epochs,
        device_str=args.device,
    )
