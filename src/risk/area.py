"""Geographic affected-area calculations."""

from typing import Optional

import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0


def cell_area_km2(latitudes, longitudes) -> torch.Tensor:
    """Return geographic cell areas using spherical latitude/longitude bounds."""
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 1 or len(lon) < 1:
        raise ValueError("latitudes and longitudes must be non-empty one-dimensional arrays")
    lat_edges = np.empty(len(lat) + 1)
    lon_edges = np.empty(len(lon) + 1)
    lat_edges[1:-1] = (lat[:-1] + lat[1:]) / 2.0
    lon_edges[1:-1] = (lon[:-1] + lon[1:]) / 2.0
    lat_edges[0] = lat[0] - (lat[1] - lat[0]) / 2.0 if len(lat) > 1 else lat[0] - 0.5
    lat_edges[-1] = lat[-1] + (lat[-1] - lat[-2]) / 2.0 if len(lat) > 1 else lat[0] + 0.5
    lon_edges[0] = lon[0] - (lon[1] - lon[0]) / 2.0 if len(lon) > 1 else lon[0] - 0.5
    lon_edges[-1] = lon[-1] + (lon[-1] - lon[-2]) / 2.0 if len(lon) > 1 else lon[0] + 0.5
    lat_band = np.abs(np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1])))
    lon_width = np.abs(np.deg2rad(lon_edges[1:] - lon_edges[:-1]))
    return torch.as_tensor((EARTH_RADIUS_KM ** 2) * lat_band[:, None] * lon_width[None, :], dtype=torch.float32)


def affected_area_km2(probability: torch.Tensor, latitudes, longitudes, probability_threshold: float = 0.5) -> torch.Tensor:
    """Sum geographic cell areas where empirical probability meets the threshold."""
    areas = cell_area_km2(latitudes, longitudes).to(device=probability.device, dtype=probability.dtype)
    mask = probability >= probability_threshold
    return (mask * areas).sum(dim=(-2, -1))
