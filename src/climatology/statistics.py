"""
Climatological statistical operations and time-aware grouping for SpatioAI.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("ClimatologyStats")


def create_time_group_key(time_coord: xr.DataArray, method: str = "dayofyear_hour") -> xr.DataArray:
    """
    Generate time-aware grouping coordinate keys handling leap years and sub-daily frequencies.

    Methods:
        - 'dayofyear_hour': Groups by day-of-year (1..366) and hour (0..23). Format: 'DOY_HH' (e.g. '001_06')
        - 'month_day_hour': Groups by calendar month, day, and hour (e.g. '0229_12' for Feb 29).
        - 'month_hour': Groups by month (1..12) and hour (0..23). Format: 'MM_HH'
        - 'dayofyear': Groups by day-of-year (1..366).
        - 'month': Groups by month (1..12).
        - 'all': Single group across all time steps.

    Args:
        time_coord: xarray time coordinate DataArray.
        method: Grouping strategy string.

    Returns:
        xr.DataArray: Grouping keys with matching 'time' dimension.
    """
    dt = time_coord.dt

    if method == "dayofyear_hour":
        # Format as string: e.g. 001_06
        doy = dt.dayofyear.values
        hour = dt.hour.values
        group_keys = [f"{d:03d}_{h:02d}" for d, h in zip(doy, hour)]
        return xr.DataArray(group_keys, coords={"time": time_coord}, dims=["time"], name="climatology_group")

    elif method == "month_day_hour":
        month = dt.month.values
        day = dt.day.values
        hour = dt.hour.values
        group_keys = [f"{m:02d}{d:02d}_{h:02d}" for m, d, h in zip(month, day, hour)]
        return xr.DataArray(group_keys, coords={"time": time_coord}, dims=["time"], name="climatology_group")

    elif method == "month_hour":
        month = dt.month.values
        hour = dt.hour.values
        group_keys = [f"{m:02d}_{h:02d}" for m, h in zip(month, hour)]
        return xr.DataArray(group_keys, coords={"time": time_coord}, dims=["time"], name="climatology_group")

    elif method == "dayofyear":
        return dt.dayofyear.rename("climatology_group")

    elif method == "month":
        return dt.month.rename("climatology_group")

    elif method == "all":
        group_keys = ["all"] * len(time_coord)
        return xr.DataArray(group_keys, coords={"time": time_coord}, dims=["time"], name="climatology_group")

    else:
        raise ValueError(f"Unsupported climatology grouping method: {method}")


def compute_group_statistics(
    da: xr.DataArray,
    group_key: xr.DataArray,
    statistics: List[str],
    dim: str = "time",
    skipna: bool = True,
) -> xr.Dataset:
    """
    Compute specified statistical metrics for a DataArray grouped by group_key.

    Supported statistics:
        - 'mean': Arithmetic average
        - 'std': Sample standard deviation
        - 'median': 50th percentile
        - 'p{N}': Percentile where N is between 0 and 100 (e.g. 'p50', 'p75', 'p90', 'p95', 'p99')

    Args:
        da: Input meteorological variable DataArray.
        group_key: Grouping DataArray (e.g. from create_time_group_key).
        statistics: List of requested statistical metric names.
        dim: Dimension to reduce (typically 'time').
        skipna: Whether to skip NaNs during computation.

    Returns:
        xr.Dataset: Dataset where each data_var is '{var_name}_{stat}'.
    """
    var_name = da.name or "variable"
    grouped = da.groupby(group_key)
    stat_arrays: Dict[str, xr.DataArray] = {}

    for stat in statistics:
        stat_lower = stat.lower()

        if stat_lower == "mean":
            res = grouped.mean(dim=dim, skipna=skipna)
            res.attrs["statistic"] = "mean"
            stat_arrays[f"{var_name}_mean"] = res

        elif stat_lower == "std":
            res = grouped.std(dim=dim, skipna=skipna)
            res.attrs["statistic"] = "std"
            stat_arrays[f"{var_name}_std"] = res

        elif stat_lower == "median":
            res = grouped.median(dim=dim, skipna=skipna)
            res.attrs["statistic"] = "median"
            stat_arrays[f"{var_name}_median"] = res

        elif stat_lower.startswith("p") and stat_lower[1:].isdigit():
            q_val = float(stat_lower[1:]) / 100.0
            res = grouped.quantile(q=q_val, dim=dim, skipna=skipna)
            # Drop the 'quantile' coordinate created by xarray to keep dataset clean
            if "quantile" in res.coords:
                res = res.drop_vars("quantile")
            res.attrs["statistic"] = f"percentile_{int(q_val * 100)}"
            stat_arrays[f"{var_name}_{stat_lower}"] = res

        else:
            logger.warning(f"Unknown or unsupported statistic '{stat}'; skipping.")

    ds_stats = xr.Dataset(stat_arrays)
    return ds_stats
