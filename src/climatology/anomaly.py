"""
Anomaly calculation engine for SpatioAI.
Supports absolute, standardized (z-score), and percentile-relative anomalies.
"""

from typing import Optional, Union
import numpy as np
import xarray as xr
from src.climatology.baseline import ClimatologyBaseline
from src.utils.logger import get_logger

logger = get_logger("Anomaly")


def compute_absolute_anomaly(
    actual: Union[xr.DataArray, xr.Dataset],
    baseline_mean: Union[xr.DataArray, xr.Dataset],
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute absolute anomaly: actual - climatological_mean.

    Args:
        actual: Observed or forecast field.
        baseline_mean: Climatological mean field.

    Returns:
        Absolute anomaly field.
    """
    anomaly = actual - baseline_mean
    if hasattr(anomaly, "attrs"):
        anomaly.attrs["anomaly_type"] = "absolute"
        anomaly.attrs["description"] = "Difference between actual and climatological mean"
    return anomaly


def compute_standardized_anomaly(
    actual: xr.DataArray,
    baseline_mean: xr.DataArray,
    baseline_std: xr.DataArray,
    eps: float = 1e-4,
    clip_range: Optional[tuple] = (-10.0, 15.0),
) -> xr.DataArray:
    """
    Compute standardized anomaly (z-score): (actual - mean) / (std + eps).
    Safely handles zero/low variance to avoid division by zero or infinite values.

    Args:
        actual: Observed or forecast DataArray.
        baseline_mean: Climatological mean DataArray.
        baseline_std: Climatological standard deviation DataArray.
        eps: Minimum standard deviation denominator regularization.
        clip_range: Optional min/max clipping to prevent unbounded artifacts.

    Returns:
        xr.DataArray: Standardized anomaly (z-scores).
    """
    # Replace zero or near-zero standard deviations with eps safely
    safe_std = xr.where(baseline_std < eps, eps, baseline_std)
    diff = actual - baseline_mean
    z_score = diff / safe_std

    if clip_range is not None:
        z_score = z_score.clip(min=clip_range[0], max=clip_range[1])

    z_score.attrs["anomaly_type"] = "standardized_z_score"
    z_score.attrs["description"] = "Standard deviations above/below climatological mean"
    return z_score


class AnomalyCalculator:
    """
    High-level interface to compute anomalies against a fitted ClimatologyBaseline.
    """

    def __init__(self, baseline: ClimatologyBaseline):
        self.baseline = baseline

    def calculate(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        method: str = "standardized",
    ) -> xr.DataArray:
        """
        Calculate anomaly for a specific variable in the dataset against the baseline.

        Args:
            ds: Input weather dataset with 'time' coord and variable.
            variable: Variable name (default: 'precipitation').
            method: Anomaly calculation method ('standardized', 'absolute', 'percentile_diff').

        Returns:
            xr.DataArray: Calculated anomaly DataArray.
        """
        if variable not in ds.data_vars:
            raise KeyError(f"Variable '{variable}' not found in dataset data_vars.")

        actual = ds[variable]
        mean_field = self.baseline.align_to_dataset(ds, stat_name="mean", variable=variable)

        if method == "absolute":
            logger.info(f"Computing absolute anomaly for '{variable}'...")
            return compute_absolute_anomaly(actual, mean_field)

        elif method == "standardized":
            logger.info(f"Computing standardized anomaly for '{variable}'...")
            std_field = self.baseline.align_to_dataset(ds, stat_name="std", variable=variable)
            return compute_standardized_anomaly(actual, mean_field, std_field)

        elif method.startswith("p") or method.startswith("percentile"):
            stat_name = method if method.startswith("p") else "p95"
            logger.info(f"Computing percentile exceedance against baseline '{stat_name}' for '{variable}'...")
            pct_field = self.baseline.align_to_dataset(ds, stat_name=stat_name, variable=variable)
            exceedance = actual - pct_field
            exceedance.attrs["anomaly_type"] = f"exceedance_{stat_name}"
            return exceedance

        else:
            raise ValueError(f"Unsupported anomaly calculation method: {method}")
