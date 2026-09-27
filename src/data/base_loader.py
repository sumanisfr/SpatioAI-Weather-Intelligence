"""
Abstract Base Weather Data Loader for SpatioAI.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xarray as xr
from src.data.preprocessing import preprocess_dataset
from src.data.quality import validate_dataset
from src.utils.logger import get_logger

logger = get_logger("BaseLoader")


class WeatherDataLoader(ABC):
    """
    Abstract Base Class for loading, filtering, standardizing, and caching meteorological data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base loader with optional configuration dictionary.
        """
        self.config = config or {}

    @abstractmethod
    def load_raw(
        self,
        source: Union[str, Path, xr.Dataset],
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Load raw data from a source (file, directory, or dataset) without modifications.
        """
        pass

    def crop_region(
        self,
        ds: xr.Dataset,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> xr.Dataset:
        """
        Crop dataset to a specific geographic bounding box.

        Args:
            ds: Input xarray Dataset.
            lat_min: Minimum latitude.
            lat_max: Maximum latitude.
            lon_min: Minimum longitude.
            lon_max: Maximum longitude.

        Returns:
            xr.Dataset: Regionally cropped dataset.
        """
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            logger.warning("Latitude/longitude coordinates not found for regional cropping.")
            return ds

        lat_vals = ds["latitude"].values
        is_lat_descending = len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]

        if is_lat_descending:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        lon_slice = slice(lon_min, lon_max)

        logger.info(f"Cropping region: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
        return ds.sel(latitude=lat_slice, longitude=lon_slice)

    def select_time_range(
        self,
        ds: xr.Dataset,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Select a slice of time if available.
        """
        if "time" not in ds.coords or (start_date is None and end_date is None):
            return ds

        logger.info(f"Selecting time slice: {start_date} to {end_date}")
        return ds.sel(time=slice(start_date, end_date))

    def select_variables(
        self,
        ds: xr.Dataset,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Select a subset of requested variables that exist in the dataset.
        """
        if not variables:
            return ds

        available_vars = [v for v in variables if v in ds.data_vars]
        missing_vars = [v for v in variables if v not in ds.data_vars]

        if missing_vars:
            logger.warning(f"Requested variables not found in dataset: {missing_vars}")

        if not available_vars:
            raise ValueError(f"None of requested variables {variables} exist in dataset data_vars: {list(ds.data_vars.keys())}")

        return ds[available_vars]

    def load(
        self,
        source: Union[str, Path, xr.Dataset],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        region: Optional[Dict[str, float]] = None,
        variables: Optional[List[str]] = None,
        validate: bool = True,
        chunks: Optional[Dict[str, int]] = None,
    ) -> xr.Dataset:
        """
        Full standardized ingestion pipeline:
        Raw Load -> Preprocess (coords, names, units) -> Crop Region -> Slice Time -> Select Variables -> Validate -> Chunk.
        """
        logger.info(f"Ingesting weather data from source: {type(source)}")
        ds = self.load_raw(source)

        # 1. Standardize coordinates & variable names
        ds = preprocess_dataset(ds)

        # 2. Slice time
        if start_date or end_date:
            ds = self.select_time_range(ds, start_date=start_date, end_date=end_date)

        # 3. Crop region
        if region:
            ds = self.crop_region(
                ds,
                lat_min=region.get("lat_min", -90.0),
                lat_max=region.get("lat_max", 90.0),
                lon_min=region.get("lon_min", -180.0),
                lon_max=region.get("lon_max", 180.0),
            )

        # 4. Select variables
        if variables:
            ds = self.select_variables(ds, variables=variables)

        # 5. Quality validation
        if validate:
            validate_dataset(ds, expected_variables=variables, raise_on_failure=True)

        # 6. Apply Dask chunks if requested
        if chunks:
            valid_chunks = {k: v for k, v in chunks.items() if k in ds.dims}
            if valid_chunks:
                logger.info(f"Applying Dask chunking: {valid_chunks}")
                ds = ds.chunk(valid_chunks)

        return ds

    @staticmethod
    def save_to_zarr(
        ds: xr.Dataset,
        zarr_path: Union[str, Path],
        mode: str = "w",
        chunks: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Save xarray Dataset to Zarr storage with chunking.
        """
        zarr_path = Path(zarr_path)
        zarr_path.parent.mkdir(parents=True, exist_ok=True)

        if chunks:
            valid_chunks = {k: v for k, v in chunks.items() if k in ds.dims}
            if valid_chunks:
                ds = ds.chunk(valid_chunks)

        logger.info(f"Writing dataset to Zarr at {zarr_path}...")
        ds.to_zarr(str(zarr_path), mode=mode, consolidated=True)
        logger.info(f"Successfully saved Zarr dataset to {zarr_path}")

    @staticmethod
    def load_from_zarr(
        zarr_path: Union[str, Path],
        chunks: Optional[Dict[str, Any]] = None,
    ) -> xr.Dataset:
        """
        Reopen a Zarr dataset lazily without loading everything into memory.
        """
        zarr_path = Path(zarr_path)
        logger.info(f"Loading Zarr dataset lazily from {zarr_path}...")
        return xr.open_zarr(str(zarr_path), chunks=chunks or "auto")
