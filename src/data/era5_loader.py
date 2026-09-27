"""
ERA5 Reanalysis Data Loader for SpatioAI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xarray as xr
from src.data.base_loader import WeatherDataLoader
from src.utils.logger import get_logger

logger = get_logger("ERA5Loader")


class ERA5Loader(WeatherDataLoader):
    """
    Loader specifically tailored for ECMWF ERA5 Reanalysis data (NetCDF / GRIB / Zarr).
    Handles ECMWF short-names (e.g. 'tp', '2t', '10u', '10v', 'sp') and unit conversions.
    """

    def load_raw(
        self,
        source: Union[str, Path, xr.Dataset],
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Load raw ERA5 dataset from path or in-memory Dataset.

        Args:
            source: File path (.nc, .grib, .zarr) or existing xarray Dataset.

        Returns:
            xr.Dataset: Raw xarray Dataset.
        """
        if isinstance(source, xr.Dataset):
            logger.info("Using provided in-memory xarray Dataset for ERA5 loader.")
            return source

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"ERA5 source path does not exist: {source_path}")

        logger.info(f"Opening ERA5 data from {source_path}")
        if source_path.suffix in [".zarr"] or source_path.is_dir():
            return xr.open_zarr(str(source_path), chunks=kwargs.get("chunks", "auto"))
        elif source_path.suffix in [".grib", ".grib2", ".grb"]:
            return xr.open_dataset(str(source_path), engine="cfgrib", chunks=kwargs.get("chunks", "auto"))
        else:
            # Default to NetCDF
            return xr.open_dataset(str(source_path), chunks=kwargs.get("chunks", "auto"))
