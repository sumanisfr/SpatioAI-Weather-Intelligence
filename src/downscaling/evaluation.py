"""
Evaluation metrics and extreme weather preservation analysis for Phase 5 Downscaling.
Evaluates:
  - General reconstruction: MAE, RMSE, Pearson Correlation, R2
  - Structural similarity: SSIM
  - Extreme weather metrics: 95th & 99th percentile bias, max intensity error (peak error),
    relative peak error, extreme spatial overlap (Critical Success Index / IoU), and spatial gradient error.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import sobel
from scipy.stats import pearsonr
from src.utils.logger import get_logger

logger = get_logger("DownscalingEvaluation")


def compute_ssim_2d(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two 2D meteorological fields.
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if data_range is None:
        data_range = max(float(np.ptp(img1)), float(np.ptp(img2)), 1e-6)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

    if denominator == 0:
        return 1.0

    return float(numerator / denominator)


def compute_gradient_error(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Gradient Error (MAGE) to quantify field sharpness and smoothing.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Sobel magnitude
    p_dy = sobel(pred, axis=0)
    p_dx = sobel(pred, axis=1)
    p_grad = np.hypot(p_dx, p_dy)

    t_dy = sobel(target, axis=0)
    t_dx = sobel(target, axis=1)
    t_grad = np.hypot(t_dx, t_dy)

    return float(np.mean(np.abs(p_grad - t_grad)))


def compute_extreme_downscaling_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    percentiles: Tuple[float, float] = (95.0, 99.0),
    extreme_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive meteorological downscaling metrics comparing prediction to target.

    Args:
        pred: Predicted 2D array or 3D/4D batch.
        target: High-resolution ground truth array.
        percentiles: Percentiles to analyze for extreme preservation (default: 95th and 99th).
        extreme_threshold: Optional physical value threshold (e.g. 50 mm) for spatial IoU/CSI.

    Returns:
        Dict containing MAE, RMSE, Pearson Correlation, SSIM, 95th/99th Bias, Max Intensity Error,
        Relative Peak Error, Extreme Area Error, and Gradient Error.
    """
    pred_arr = np.asarray(pred, dtype=np.float64).ravel()
    tgt_arr = np.asarray(target, dtype=np.float64).ravel()

    # 1. Standard Reconstruction Metrics
    mae = float(np.mean(np.abs(pred_arr - tgt_arr)))
    mse = float(np.mean((pred_arr - tgt_arr) ** 2))
    rmse = float(np.sqrt(mse))

    # Pearson Correlation
    if np.std(pred_arr) > 1e-6 and np.std(tgt_arr) > 1e-6:
        r_val, _ = pearsonr(pred_arr, tgt_arr)
        corr = float(r_val) if not np.isnan(r_val) else 0.0
    else:
        corr = 1.0 if np.allclose(pred_arr, tgt_arr) else 0.0

    # R-squared
    ss_tot = float(np.sum((tgt_arr - np.mean(tgt_arr)) ** 2))
    ss_res = float(np.sum((tgt_arr - pred_arr) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 1.0

    # 2. Structural Quality (SSIM)
    if pred.ndim >= 2 and target.ndim >= 2:
        if pred.ndim == 2:
            ssim_val = compute_ssim_2d(pred, target)
            grad_err = compute_gradient_error(pred, target)
        else:
            # Multi-sample batch average
            pred_2d_list = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
            tgt_2d_list = target.reshape(-1, target.shape[-2], target.shape[-1])
            ssim_vals = [compute_ssim_2d(p, t) for p, t in zip(pred_2d_list, tgt_2d_list)]
            grad_errs = [compute_gradient_error(p, t) for p, t in zip(pred_2d_list, tgt_2d_list)]
            ssim_val = float(np.mean(ssim_vals))
            grad_err = float(np.mean(grad_errs))
    else:
        ssim_val = 1.0
        grad_err = 0.0

    # 3. Extreme Percentile Biases
    p95_pred = float(np.percentile(pred_arr, percentiles[0]))
    p95_tgt = float(np.percentile(tgt_arr, percentiles[0]))
    bias_p95 = p95_pred - p95_tgt

    p99_pred = float(np.percentile(pred_arr, percentiles[1]))
    p99_tgt = float(np.percentile(tgt_arr, percentiles[1]))
    bias_p99 = p99_pred - p99_tgt

    # 4. Maximum Intensity (Peak) Errors
    max_pred = float(np.max(pred_arr))
    max_tgt = float(np.max(tgt_arr))
    peak_error = max_pred - max_tgt
    rel_peak_error = abs(peak_error) / max_tgt if max_tgt > 1e-6 else 0.0

    # 5. Extreme Spatial Overlap (Critical Success Index / IoU)
    if extreme_threshold is None:
        # Use target 95th percentile as default extreme threshold
        extreme_threshold = p95_tgt

    pred_mask = pred_arr >= extreme_threshold
    tgt_mask = tgt_arr >= extreme_threshold

    hits = int(np.sum(pred_mask & tgt_mask))
    misses = int(np.sum((~pred_mask) & tgt_mask))
    false_alarms = int(np.sum(pred_mask & (~tgt_mask)))

    denom = hits + misses + false_alarms
    csi_extreme = float(hits / denom) if denom > 0 else 1.0

    area_pred = int(np.sum(pred_mask))
    area_tgt = int(np.sum(tgt_mask))
    area_error = area_pred - area_tgt

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "correlation": round(corr, 4),
        "r2": round(r2, 4),
        "ssim": round(ssim_val, 4),
        "bias_p95": round(bias_p95, 4),
        "bias_p99": round(bias_p99, 4),
        "peak_error": round(peak_error, 4),
        "rel_peak_error": round(rel_peak_error, 4),
        "csi_extreme": round(csi_extreme, 4),
        "area_error": int(area_error),
        "gradient_error": round(grad_err, 4),
        "max_pred": round(max_pred, 2),
        "max_target": round(max_tgt, 2),
    }


def compare_downscaling_methods(
    predictions_dict: Dict[str, np.ndarray],
    target: np.ndarray,
    percentiles: Tuple[float, float] = (95.0, 99.0),
) -> Dict[str, Dict[str, float]]:
    """
    Generate side-by-side metric comparison across multiple downscaling models/baselines.

    Args:
        predictions_dict: Dict mapping method name (e.g. 'Nearest', 'Bilinear', 'UNet') to predicted array.
        target: Ground truth high-resolution array.

    Returns:
        Dict mapping method name to its metrics dictionary.
    """
    results = {}
    for method_name, pred_arr in predictions_dict.items():
        metrics = compute_extreme_downscaling_metrics(pred_arr, target, percentiles=percentiles)
        results[method_name] = metrics
    return results
