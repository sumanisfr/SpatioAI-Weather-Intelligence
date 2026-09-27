"""
Regridding module for spatial interpolation and grid harmonization.
"""

from typing import Optional, Tuple
import numpy as np
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("Regridding")


class Regridder:
    """
    Modular regridder for interpolating weather fields onto target spatial grids.
    Supports bilinear and nearest neighbor interpolation with xarray.
    """

    def __init__(
        self,
        target_lat: Optional[np.ndarray] = None,
        target_lon: Optional[np.ndarray] = None,
        lat_step: Optional[float] = None,
        lon_step: Optional[float] = None,
        lat_bounds: Optional[Tuple[float, float]] = None,
        lon_bounds: Optional[Tuple[float, float]] = None,
        method: str = "linear",
    ):
        """
        Initialize regridder with target coordinates or bounding box + resolution.

        Args:
            target_lat: 1D array of target latitude coordinates.
            target_lon: 1D array of target longitude coordinates.
            lat_step: Grid step for latitude in degrees (e.g., 0.1 for ~12km, 0.045 for ~5km).
            lon_step: Grid step for longitude in degrees.
            lat_bounds: Tuple of (min_lat, max_lat).
            lon_bounds: Tuple of (min_lon, max_lon).
            method: Interpolation method ('linear', 'nearest', 'cubic').
        """
        self.method = method

        if target_lat is not None and target_lon is not None:
            self.target_lat = target_lat
            self.target_lon = target_lon
        elif lat_bounds is not None and lon_bounds is not None and lat_step is not None:
            lon_step = lon_step or lat_step
            self.target_lat = np.arange(lat_bounds[0], lat_bounds[1] + lat_step * 0.5, lat_step)
            self.target_lon = np.arange(lon_bounds[0], lon_bounds[1] + lon_step * 0.5, lon_step)
        else:
            self.target_lat = None
            self.target_lon = None

    def regrid(self, ds: xr.Dataset, method: Optional[str] = None) -> xr.Dataset:
        """
        Interpolate the dataset to the target spatial grid.

        Args:
            ds: Source xarray Dataset with 'latitude' and 'longitude' coordinates.
            method: Optional override for interpolation method.

        Returns:
            xr.Dataset: Regridded dataset.
        """
        if self.target_lat is None or self.target_lon is None:
            logger.info("No target grid defined; skipping regridding.")
            return ds

        interp_method = method or self.method

        logger.info(
            f"Regridding dataset from shape ({len(ds.get('latitude', []))}, {len(ds.get('longitude', []))}) "
            f"to target grid ({len(self.target_lat)}, {len(self.target_lon)}) using '{interp_method}' interpolation."
        )

        # Use xarray's interp for lazy and multi-dimensional interpolation
        regridded_ds = ds.interp(
            latitude=self.target_lat,
            longitude=self.target_lon,
            method=interp_method,
        )

        regridded_ds.attrs["regrid_method"] = interp_method
        regridded_ds.attrs["regrid_target_shape"] = f"{len(self.target_lat)}x{len(self.target_lon)}"

        return regridded_ds
