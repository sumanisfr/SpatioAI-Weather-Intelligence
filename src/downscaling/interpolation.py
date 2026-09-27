"""
Classical spatial interpolation baselines (Nearest-neighbor, Bilinear, Bicubic) for weather downscaling.
Independent from PyTorch where possible; operates on standard numpy arrays and scipy/xarray grids.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.utils.logger import get_logger

logger = get_logger("DownscalingInterpolation")


def interpolate_field(
    field: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "bilinear",
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Spatially interpolate a 2D or 3D weather field from a coarse source grid to a fine target grid.

    Args:
        field: numpy array of shape [n_src_lat, n_src_lon] or [batch/time, n_src_lat, n_src_lon]
               or [batch, channel, n_src_lat, n_src_lon].
        source_lat: 1D array of source latitude coordinates.
        source_lon: 1D array of source longitude coordinates.
        target_lat: 1D array of target high-resolution latitude coordinates.
        target_lon: 1D array of target high-resolution longitude coordinates.
        method: Interpolation method: 'nearest', 'bilinear' (linear), 'bicubic' (cubic), or 'slinear'.
        fill_value: Fill value for extrapolation outside source bounds. If None, extrapolates with boundary value.

    Returns:
        numpy array interpolated onto [target_lat, target_lon] with corresponding batch/channel leading dims.
    """
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)

    # Scipy RegularGridInterpolator method mapping
    method_map = {
        "nearest": "nearest",
        "bilinear": "linear",
        "linear": "linear",
        "bicubic": "cubic",
        "cubic": "cubic",
    }
    scipy_method = method_map.get(method.lower(), "linear")

    # Ensure source coordinate arrays are strictly ascending for RegularGridInterpolator
    lat_flip = False
    lon_flip = False

    if len(source_lat) > 1 and source_lat[1] < source_lat[0]:
        source_lat = np.flip(source_lat)
        lat_flip = True

    if len(source_lon) > 1 and source_lon[1] < source_lon[0]:
        source_lon = np.flip(source_lon)
        lon_flip = True

    # Target meshgrid
    tgt_mesh_lat, tgt_mesh_lon = np.meshgrid(target_lat, target_lon, indexing="ij")
    target_points = np.stack([tgt_mesh_lat.ravel(), tgt_mesh_lon.ravel()], axis=-1)

    field_arr = np.asarray(field, dtype=np.float64)

    # Handle 2D: [lat, lon]
    if field_arr.ndim == 2:
        src_data = field_arr
        if lat_flip:
            src_data = np.flip(src_data, axis=0)
        if lon_flip:
            src_data = np.flip(src_data, axis=1)

        interp_func = RegularGridInterpolator(
            (source_lat, source_lon),
            src_data,
            method=scipy_method,
            bounds_error=False,
            fill_value=fill_value,
        )
        out_flat = interp_func(target_points)
        out_2d = out_flat.reshape(len(target_lat), len(target_lon))

        # Handle NaNs from extrapolation if fill_value is None
        if np.any(np.isnan(out_2d)):
            # Nearest fallback for boundary NaNs
            nearest_func = RegularGridInterpolator(
                (source_lat, source_lon),
                src_data,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )
            nan_mask = np.isnan(out_2d)
            out_2d[nan_mask] = nearest_func(target_points).reshape(len(target_lat), len(target_lon))[nan_mask]

        return out_2d

    # Handle 3D: [B, lat, lon]
    elif field_arr.ndim == 3:
        B = field_arr.shape[0]
        out_3d = np.zeros((B, len(target_lat), len(target_lon)), dtype=field_arr.dtype)
        for b in range(B):
            out_3d[b] = interpolate_field(
                field=field_arr[b],
                source_lat=source_lat if not lat_flip else np.flip(source_lat),
                source_lon=source_lon if not lon_flip else np.flip(source_lon),
                target_lat=target_lat,
                target_lon=target_lon,
                method=method,
                fill_value=fill_value,
            )
        return out_3d

    # Handle 4D: [B, C, lat, lon]
    elif field_arr.ndim == 4:
        B, C = field_arr.shape[:2]
        out_4d = np.zeros((B, C, len(target_lat), len(target_lon)), dtype=field_arr.dtype)
        for b in range(B):
            for c in range(C):
                out_4d[b, c] = interpolate_field(
                    field=field_arr[b, c],
                    source_lat=source_lat if not lat_flip else np.flip(source_lat),
                    source_lon=source_lon if not lon_flip else np.flip(source_lon),
                    target_lat=target_lat,
                    target_lon=target_lon,
                    method=method,
                    fill_value=fill_value,
                )
        return out_4d

    else:
        raise ValueError(f"Unsupported field array dimension: {field_arr.ndim}")
