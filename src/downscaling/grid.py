"""
Grid calculation and dynamic spatial event cropping utilities for SpatioAI Phase 5 downscaling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("DownscalingGrid")

# 1 degree latitude is approximately 111.139 km on WGS84 sphere
KM_PER_DEGREE_LAT = 111.139


def calculate_grid_spacing(
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate effective spatial resolution and grid spacing in degrees and kilometers.
    Accounts for longitude grid convergence with latitude (cos(lat)).

    Args:
        lats: 1D array of latitude coordinates in degrees.
        lons: 1D array of longitude coordinates in degrees.

    Returns:
        Dict with:
            - dlat_deg: Delta latitude in degrees
            - dlon_deg: Delta longitude in degrees
            - dlat_km: Latitude grid spacing in km
            - dlon_km_mean: Mean longitude grid spacing in km across region
            - effective_res_km: Geometric mean effective resolution in km
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    if len(lats) < 2 or len(lons) < 2:
        return {
            "dlat_deg": 0.0,
            "dlon_deg": 0.0,
            "dlat_km": 0.0,
            "dlon_km_mean": 0.0,
            "effective_res_km": 0.0,
        }

    dlat_deg = float(np.abs(np.mean(np.diff(lats))))
    dlon_deg = float(np.abs(np.mean(np.diff(lons))))

    dlat_km = dlat_deg * KM_PER_DEGREE_LAT
    mean_lat_rad = np.radians(np.mean(lats))
    dlon_km_mean = dlon_deg * KM_PER_DEGREE_LAT * float(np.cos(mean_lat_rad))
    effective_res_km = float(np.sqrt(dlat_km * dlon_km_mean))

    return {
        "dlat_deg": round(dlat_deg, 5),
        "dlon_deg": round(dlon_deg, 5),
        "dlat_km": round(dlat_km, 3),
        "dlon_km_mean": round(dlon_km_mean, 3),
        "effective_res_km": round(effective_res_km, 3),
    }


def create_target_grid(
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    target_res_km: float = 5.0,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a high-resolution target grid (lats, lons) for downscaling.

    Args:
        lat_bounds: (min_lat, max_lat) in degrees.
        lon_bounds: (min_lon, max_lon) in degrees.
        target_res_km: Approximate target grid spacing in kilometers (default ~5 km).
        target_shape: Optional explicit (n_lats, n_lons) grid dimensions.

    Returns:
        Tuple[target_lats, target_lons] 1D numpy arrays.
    """
    min_lat, max_lat = min(lat_bounds), max(lat_bounds)
    min_lon, max_lon = min(lon_bounds), max(lon_bounds)

    if target_shape is not None:
        n_lat, n_lon = target_shape
        target_lats = np.linspace(min_lat, max_lat, n_lat, dtype=np.float64)
        target_lons = np.linspace(min_lon, max_lon, n_lon, dtype=np.float64)
        return target_lats, target_lons

    # Compute step sizes in degrees based on target resolution in km
    mean_lat_rad = np.radians((min_lat + max_lat) / 2.0)
    cos_lat = max(float(np.cos(mean_lat_rad)), 0.1)

    dlat_deg = target_res_km / KM_PER_DEGREE_LAT
    dlon_deg = target_res_km / (KM_PER_DEGREE_LAT * cos_lat)

    n_lat = max(4, int(np.round((max_lat - min_lat) / dlat_deg)) + 1)
    n_lon = max(4, int(np.round((max_lon - min_lon) / dlon_deg)) + 1)

    target_lats = np.linspace(min_lat, max_lat, n_lat, dtype=np.float64)
    target_lons = np.linspace(min_lon, max_lon, n_lon, dtype=np.float64)

    return target_lats, target_lons


def extract_event_crop(
    field_or_ds: Union[np.ndarray, xr.DataArray, xr.Dataset],
    bbox: Tuple[float, float, float, float],
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    padding_deg: float = 1.0,
    timestamp: Optional[Any] = None,
    variable: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dynamically extract a spatial crop bounding an extreme weather event with padding.

    Args:
        field_or_ds: 2D/3D numpy array, xr.DataArray, or xr.Dataset.
        bbox: (min_lat, max_lat, min_lon, max_lon) bounding box in degrees.
        lats: Latitude coordinates (required if field_or_ds is numpy array).
        lons: Longitude coordinates (required if field_or_ds is numpy array).
        padding_deg: Spatial buffer around bounding box in degrees.
        timestamp: Optional timestamp selector if dataset contains time dimension.
        variable: Variable name if passing xr.Dataset.

    Returns:
        Dict containing:
            - 'crop_data': 2D numpy array [n_lat, n_lon] or 3D [time, n_lat, n_lon]
            - 'lats': 1D array of cropped latitude coordinates
            - 'lons': 1D array of cropped longitude coordinates
            - 'bbox_padded': (crop_min_lat, crop_max_lat, crop_min_lon, crop_max_lon)
            - 'grid_spacing': Dict of grid metrics
            - 'metadata': Contextual dictionary
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    # Apply padding
    crop_min_lat = min_lat - padding_deg
    crop_max_lat = max_lat + padding_deg
    crop_min_lon = min_lon - padding_deg
    crop_max_lon = max_lon + padding_deg

    # Extract coordinates and data
    if isinstance(field_or_ds, (xr.Dataset, xr.DataArray)):
        src = field_or_ds
        if isinstance(src, xr.Dataset):
            var_name = variable or list(src.data_vars.keys())[0]
            da = src[var_name]
        else:
            da = src

        if timestamp is not None and "time" in da.dims:
            da = da.sel(time=timestamp, method="nearest")

        all_lats = da["latitude"].values if "latitude" in da.coords else da["lat"].values
        all_lons = da["longitude"].values if "longitude" in da.coords else da["lon"].values

        # Determine index slices
        lat_mask = (all_lats >= crop_min_lat) & (all_lats <= crop_max_lat)
        lon_mask = (all_lons >= crop_min_lon) & (all_lons <= crop_max_lon)

        if not np.any(lat_mask) or not np.any(lon_mask):
            # Fallback to closest available bounds
            lat_idx_min = np.clip(np.searchsorted(np.sort(all_lats), crop_min_lat), 0, len(all_lats) - 1)
            lat_idx_max = np.clip(np.searchsorted(np.sort(all_lats), crop_max_lat), 0, len(all_lats) - 1)
            lon_idx_min = np.clip(np.searchsorted(np.sort(all_lons), crop_min_lon), 0, len(all_lons) - 1)
            lon_idx_max = np.clip(np.searchsorted(np.sort(all_lons), crop_max_lon), 0, len(all_lons) - 1)
            crop_da = da.isel(
                latitude=slice(min(lat_idx_min, lat_idx_max), max(lat_idx_min, lat_idx_max) + 1),
                longitude=slice(min(lon_idx_min, lon_idx_max), max(lon_idx_min, lon_idx_max) + 1),
            )
        else:
            lat_dim = "latitude" if "latitude" in da.dims else "lat"
            lon_dim = "longitude" if "longitude" in da.dims else "lon"
            crop_da = da.sel({lat_dim: slice(crop_min_lat, crop_max_lat), lon_dim: slice(crop_min_lon, crop_max_lon)})

        crop_lats = crop_da[lat_dim].values
        crop_lons = crop_da[lon_dim].values
        crop_data = crop_da.values

    else:
        if lats is None or lons is None:
            raise ValueError("lats and lons must be provided when field is a numpy array.")
        all_lats = np.asarray(lats)
        all_lons = np.asarray(lons)

        lat_indices = np.where((all_lats >= crop_min_lat) & (all_lats <= crop_max_lat))[0]
        lon_indices = np.where((all_lons >= crop_min_lon) & (all_lons <= crop_max_lon))[0]

        if len(lat_indices) == 0:
            lat_indices = np.array([np.argmin(np.abs(all_lats - (min_lat + max_lat) / 2.0))])
        if len(lon_indices) == 0:
            lon_indices = np.array([np.argmin(np.abs(all_lons - (min_lon + max_lon) / 2.0))])

        crop_lats = all_lats[lat_indices[0] : lat_indices[-1] + 1]
        crop_lons = all_lons[lon_indices[0] : lon_indices[-1] + 1]

        if field_or_ds.ndim == 2:
            crop_data = field_or_ds[lat_indices[0] : lat_indices[-1] + 1, lon_indices[0] : lon_indices[-1] + 1]
        elif field_or_ds.ndim == 3:
            crop_data = field_or_ds[:, lat_indices[0] : lat_indices[-1] + 1, lon_indices[0] : lon_indices[-1] + 1]
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {field_or_ds.ndim}")

    grid_spacing = calculate_grid_spacing(crop_lats, crop_lons)

    return {
        "crop_data": crop_data,
        "lats": crop_lats,
        "lons": crop_lons,
        "bbox_padded": (float(crop_lats.min()), float(crop_lats.max()), float(crop_lons.min()), float(crop_lons.max())),
        "grid_spacing": grid_spacing,
        "metadata": {
            "orig_bbox": bbox,
            "padding_deg": padding_deg,
            "shape": crop_data.shape,
            "timestamp": str(timestamp) if timestamp is not None else None,
        },
    }
