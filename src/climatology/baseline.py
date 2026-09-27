"""
Climatological baseline builder and manager for SpatioAI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xarray as xr
from src.climatology.statistics import compute_group_statistics, create_time_group_key
from src.utils.logger import get_logger

logger = get_logger("ClimatologyBaseline")


class ClimatologyBaseline:
    """
    Computes, stores, and aligns multi-variable historical climatological baselines.
    """

    def __init__(
        self,
        grouping_method: str = "dayofyear_hour",
        variable_configs: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ):
        """
        Args:
            grouping_method: Time grouping method ('dayofyear_hour', 'month_hour', etc.).
            variable_configs: Dictionary mapping variable names to their desired statistics, e.g.:
                {'precipitation': {'statistics': ['mean', 'std', 'p90', 'p95', 'p99']}}
        """
        self.grouping_method = grouping_method
        self.variable_configs = variable_configs or {
            "precipitation": {"statistics": ["mean", "std", "median", "p90", "p95", "p99"]},
            "temperature": {"statistics": ["mean", "std", "p90", "p95"]},
        }
        self.baseline_ds: Optional[xr.Dataset] = None

    def fit(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Calculate climatological baseline statistics from an input historical dataset.

        Args:
            ds: Historical xarray.Dataset.

        Returns:
            xr.Dataset: Climatology baseline dataset indexed by climatology_group.
        """
        if "time" not in ds.coords:
            raise ValueError("Historical dataset must contain a 'time' coordinate.")

        logger.info(
            f"Fitting climatology baseline across {len(ds.time)} timestamps "
            f"using '{self.grouping_method}' grouping..."
        )

        group_key = create_time_group_key(ds["time"], method=self.grouping_method)

        stat_datasets: List[xr.Dataset] = []

        for var_name, var_cfg in self.variable_configs.items():
            if var_name not in ds.data_vars:
                logger.warning(f"Variable '{var_name}' not found in dataset data_vars; skipping baseline fit.")
                continue

            requested_stats = var_cfg.get("statistics", ["mean", "std"])
            logger.info(f"Computing statistics {requested_stats} for '{var_name}'...")

            var_stats = compute_group_statistics(
                da=ds[var_name],
                group_key=group_key,
                statistics=requested_stats,
            )
            stat_datasets.append(var_stats)

        if not stat_datasets:
            raise ValueError("No matching variables found to compute climatology baseline.")

        self.baseline_ds = xr.merge(stat_datasets)
        self.baseline_ds.attrs["grouping_method"] = self.grouping_method
        self.baseline_ds.attrs["description"] = "SpatioAI Climatological Baseline"

        logger.info(f"Climatology baseline fit complete with variables: {list(self.baseline_ds.data_vars.keys())}")
        return self.baseline_ds

    def save(self, output_path: Union[str, Path], mode: str = "w") -> None:
        """
        Save baseline dataset to Zarr.
        """
        if self.baseline_ds is None:
            raise ValueError("No baseline fitted yet. Call fit() or load() before saving.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving climatology baseline to Zarr at {path}...")
        self.baseline_ds.to_zarr(str(path), mode=mode, consolidated=True)
        logger.info("Climatology baseline successfully saved.")

    def load(self, zarr_path: Union[str, Path]) -> xr.Dataset:
        """
        Load a fitted climatology baseline dataset from Zarr.
        """
        path = Path(zarr_path)
        if not path.exists():
            raise FileNotFoundError(f"Climatology baseline store not found at {path}")

        logger.info(f"Loading climatology baseline from Zarr: {path}...")
        self.baseline_ds = xr.open_zarr(str(path))
        self.grouping_method = self.baseline_ds.attrs.get("grouping_method", self.grouping_method)
        return self.baseline_ds

    def align_to_dataset(self, target_ds: xr.Dataset, stat_name: str, variable: str = "precipitation") -> xr.DataArray:
        """
        Align the climatology baseline metric for a given variable with the timestamps in target_ds.

        Args:
            target_ds: Target weather dataset (e.g. forecast or new observation) with 'time' coord.
            stat_name: Statistic name (e.g. 'mean', 'std', 'p95', 'p99').
            variable: Variable name (e.g. 'precipitation').

        Returns:
            xr.DataArray: Baseline field aligned with target_ds time, latitude, and longitude.
        """
        if self.baseline_ds is None:
            raise ValueError("Climatology baseline is not loaded or fitted.")

        field_name = f"{variable}_{stat_name}"
        if field_name not in self.baseline_ds.data_vars:
            raise KeyError(
                f"Climatology field '{field_name}' not found. Available: {list(self.baseline_ds.data_vars.keys())}"
            )

        # Generate grouping keys for target dataset timestamps
        target_group_keys = create_time_group_key(target_ds["time"], method=self.grouping_method)

        # Baseline DataArray indexed by 'climatology_group'
        base_da = self.baseline_ds[field_name]

        # Map target timestamps to the corresponding baseline group slices
        aligned_slices = []
        for grp in target_group_keys.values:
            if grp in base_da["climatology_group"].values:
                aligned_slices.append(base_da.sel(climatology_group=grp))
            else:
                # Fallback: if specific group key missing, pick closest or overall mean
                logger.warning(f"Group key '{grp}' not found in baseline; falling back to average across groups.")
                aligned_slices.append(base_da.mean(dim="climatology_group"))

        aligned_da = xr.concat(aligned_slices, dim="time")
        aligned_da = aligned_da.assign_coords(time=target_ds["time"])
        return aligned_da
