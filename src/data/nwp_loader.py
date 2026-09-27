"""
Generic Numerical Weather Prediction (NWP) Data Loader for SpatioAI.
Supports models like NCUM, NEPS-G, GFS, and others.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xarray as xr
from src.data.base_loader import WeatherDataLoader
from src.utils.logger import get_logger

logger = get_logger("NWPLoader")


class NWPLoader(WeatherDataLoader):
    """
    Generic NWP loader handling multi-lead-time forecast datasets.
    """

    def load_raw(
        self,
        source: Union[str, Path, xr.Dataset],
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Load raw NWP dataset from file or in-memory object.

        Args:
            source: Path or Dataset.

        Returns:
            xr.Dataset: Raw NWP xarray Dataset.
        """
        if isinstance(source, xr.Dataset):
            logger.info("Using provided in-memory xarray Dataset for NWP loader.")
            return source

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"NWP source path does not exist: {source_path}")

        logger.info(f"Opening NWP data from {source_path}")
        if source_path.suffix in [".zarr"] or source_path.is_dir():
            return xr.open_zarr(str(source_path), chunks=kwargs.get("chunks", "auto"))
        elif source_path.suffix in [".grib", ".grib2", ".grb"]:
            return xr.open_dataset(str(source_path), engine="cfgrib", chunks=kwargs.get("chunks", "auto"))
        else:
            return xr.open_dataset(str(source_path), chunks=kwargs.get("chunks", "auto"))

    def select_forecast_horizon(
        self,
        ds: xr.Dataset,
        lead_time_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Select specific forecast step or lead time if present.
        """
        for lead_dim in ["step", "lead_time", "forecast_period"]:
            if lead_dim in ds.dims and lead_time_hours is not None:
                logger.info(f"Selecting lead times along '{lead_dim}': {lead_time_hours}")
                return ds.sel({lead_dim: lead_time_hours})
        return ds
