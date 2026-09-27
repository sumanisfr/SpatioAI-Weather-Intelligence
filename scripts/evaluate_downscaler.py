"""
Evaluation and baseline comparison script for Phase 5 Downscaling.
Compares:
  - Nearest-Neighbor Baseline
  - Bilinear Interpolation Baseline
  - U-Net Downscaler
Computes full metrics table, extreme-event preservation statistics, and outputs summary table and figures.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.downscaling.dataset import DownscalingDataset
from src.downscaling.evaluation import compare_downscaling_methods
from src.downscaling.inference import DownscalingPredictor
from src.downscaling.synthetic import generate_paired_synthetic_dataset
from src.utils.logger import get_logger

logger = get_logger("EvaluateDownscaler")


def evaluate_downscaling_pipeline(
    config_path: str = "configs/data.yaml",
    checkpoint_path: Optional[str] = None,
    save_plots: bool = True,
) -> pd.DataFrame:
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("downscaling", {})
    paths_cfg = ds_cfg.get("paths", {})
    ckpt = checkpoint_path or paths_cfg.get("model_checkpoint", "models/downscaler_unet_best.pt")
    output_dir = Path(paths_cfg.get("output_dir", "data/outputs/downscaling"))
    figs_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    preds_dir = output_dir / "predictions"

    for d in [figs_dir, metrics_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load test split
    test_pkl = Path(paths_cfg.get("dataset_dir", "data/downscaling")) / "test" / "samples.pkl"
    if test_pkl.exists():
        with open(test_pkl, "rb") as f:
            test_samples = pickle.load(f)
    else:
        logger.warning("Test dataset not found. Generating on the fly...")
        test_samples = generate_paired_synthetic_dataset(num_events=6, timesteps_per_event=4, seed=123)

    logger.info(f"Evaluating across {len(test_samples)} held-out test samples...")

    # Load Predictor
    predictor = DownscalingPredictor(checkpoint_path=ckpt)

    all_nearest = []
    all_bilinear = []
    all_unet = []
    all_targets = []

    for i, item in enumerate(test_samples):
        res = predictor.predict_field(
            low_res_field=item["low_res"],
            source_lats=item["low_lats"],
            source_lons=item["low_lons"],
            target_lats=item["high_lats"],
            target_lons=item["high_lons"],
        )
        all_nearest.append(res["baseline_nearest"])
        all_bilinear.append(res["baseline_bilinear"])
        all_unet.append(res["prediction"])
        all_targets.append(item["high_res"])

    all_nearest_arr = np.stack(all_nearest, axis=0)
    all_bilinear_arr = np.stack(all_bilinear, axis=0)
    all_unet_arr = np.stack(all_unet, axis=0)
    all_targets_arr = np.stack(all_targets, axis=0)

    # Comparison metrics
    predictions = {
        "Nearest": all_nearest_arr,
        "Bilinear": all_bilinear_arr,
        "U-Net": all_unet_arr,
    }

    comparison = compare_downscaling_methods(predictions, all_targets_arr, percentiles=(95.0, 99.0))
    df_metrics = pd.DataFrame(comparison).T
    df_metrics = df_metrics[["mae", "rmse", "correlation", "ssim", "bias_p95", "bias_p99", "peak_error", "rel_peak_error", "csi_extreme", "gradient_error"]]

    # Print Table
    print("\n" + "=" * 95)
    print("               SPATIOAI PHASE 5: DOWNSCALING MODEL COMPARISON TABLE")
    print("=" * 95)
    print(df_metrics.to_string())
    print("=" * 95 + "\n")

    # Save metrics
    metrics_file = metrics_dir / "downscaling_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(comparison, f, indent=2)
    df_metrics.to_csv(metrics_dir / "downscaling_comparison.csv")
    logger.info(f"Saved evaluation metrics to {metrics_file}")

    # Generate sample visualization figure
    if save_plots and len(test_samples) > 0:
        sample_idx = 0
        sample_item = test_samples[sample_idx]
        low_res = sample_item["low_res"]
        target = sample_item["high_res"]
        pred_res = predictor.predict_field(
            low_res_field=low_res,
            source_lats=sample_item["low_lats"],
            source_lons=sample_item["low_lons"],
            target_lats=sample_item["high_lats"],
            target_lons=sample_item["high_lons"],
        )

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        vmax = max(float(np.max(target)), float(np.max(pred_res["prediction"])), 10.0)

        # 1. Coarse Input
        im0 = axes[0, 0].imshow(low_res, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        axes[0, 0].set_title(f"Coarse ~12km Input\n(Max: {np.max(low_res):.1f} mm)")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Bilinear Baseline
        im1 = axes[0, 1].imshow(pred_res["baseline_bilinear"], origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        axes[0, 1].set_title(f"Bilinear Baseline\n(Max: {np.max(pred_res['baseline_bilinear']):.1f} mm)")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. U-Net Prediction
        im2 = axes[0, 2].imshow(pred_res["prediction"], origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        axes[0, 2].set_title(f"U-Net ~5km Downscaled\n(Max: {np.max(pred_res['prediction']):.1f} mm)")
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 4. Target Ground Truth
        im3 = axes[1, 0].imshow(target, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        axes[1, 0].set_title(f"Target Ground Truth\n(Max: {np.max(target):.1f} mm)")
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 5. Bilinear Error Map
        bilinear_err = pred_res["baseline_bilinear"] - target
        max_err = max(float(np.max(np.abs(bilinear_err))), 1.0)
        im4 = axes[1, 1].imshow(bilinear_err, origin="lower", cmap="coolwarm", vmin=-max_err, vmax=max_err)
        axes[1, 1].set_title(f"Bilinear Error (MAE: {np.mean(np.abs(bilinear_err)):.2f} mm)")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 6. U-Net Error Map
        unet_err = pred_res["prediction"] - target
        im5 = axes[1, 2].imshow(unet_err, origin="lower", cmap="coolwarm", vmin=-max_err, vmax=max_err)
        axes[1, 2].set_title(f"U-Net Error (MAE: {np.mean(np.abs(unet_err)):.2f} mm)")
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.suptitle("SpatioAI Phase 5 Extreme-Event Downscaling (12km -> ~5km)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        plot_path = figs_dir / "downscaling_sample_comparison.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved visualization figure to {plot_path}")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Downscaler")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    evaluate_downscaling_pipeline(config_path=args.config, checkpoint_path=args.checkpoint)
