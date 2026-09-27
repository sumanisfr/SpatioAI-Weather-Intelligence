"""
Synthetic weather dataset generator for SpatioAI Phase 1 and Phase 2 testing.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr


def generate_synthetic_weather_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-05",
    freq: str = "6h",
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lat_step: float = 0.5,
    lon_min: float = 65.0,
    lon_max: float = 100.0,
    lon_step: float = 0.5,
    variables: Optional[List[str]] = None,
    seed: int = 42,
) -> xr.Dataset:
    """
    Generate a synthetic multi-variable xarray weather dataset with realistic ranges.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        freq: Time frequency (e.g. '6h', '1h', 'D').
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lat_step: Latitude grid spacing in degrees.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        lon_step: Longitude grid spacing in degrees.
        variables: List of variable names to include.
        seed: Random seed for reproducibility.

    Returns:
        xr.Dataset: Synthetic weather dataset with coordinates and metadata.
    """
    if variables is None:
        variables = ["precipitation", "temperature", "u10", "v10", "surface_pressure"]

    rng = np.random.default_rng(seed)

    time_coords = pd.date_range(start=start_date, end=end_date, freq=freq)
    lat_coords = np.arange(lat_min, lat_max + lat_step * 0.5, lat_step)
    lon_coords = np.arange(lon_min, lon_max + lon_step * 0.5, lon_step)

    n_time = len(time_coords)
    n_lat = len(lat_coords)
    n_lon = len(lon_coords)
    shape = (n_time, n_lat, n_lon)

    data_vars: Dict[str, tuple] = {}

    for var in variables:
        if var in ["precipitation", "tp"]:
            # Normal background precipitation: exponential distributed with small mean
            base_precip = rng.exponential(scale=3.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                base_precip.astype(np.float32),
                {"units": "mm", "long_name": "Total Precipitation"},
            )

        elif var in ["temperature", "t2m", "2t"]:
            # Tropical / Subtropical temperatures (280 K - 315 K / ~7 C - 42 C)
            base_temp = 295.0 + rng.normal(loc=0.0, scale=4.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                base_temp.astype(np.float32),
                {"units": "K", "long_name": "2-Metre Temperature"},
            )

        elif var in ["u10", "u_wind"]:
            # Zonal wind component (-25 m/s to 25 m/s)
            u_wind = rng.normal(loc=2.0, scale=6.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                u_wind.astype(np.float32),
                {"units": "m/s", "long_name": "10-Metre U Wind Component"},
            )

        elif var in ["v10", "v_wind"]:
            # Meridional wind component (-25 m/s to 25 m/s)
            v_wind = rng.normal(loc=-1.0, scale=6.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                v_wind.astype(np.float32),
                {"units": "m/s", "long_name": "10-Metre V Wind Component"},
            )

        elif var in ["surface_pressure", "sp"]:
            # Surface pressure (980 hPa to 1025 hPa)
            sp = 1013.25 + rng.normal(loc=0.0, scale=6.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                sp.astype(np.float32),
                {"units": "hPa", "long_name": "Surface Pressure"},
            )

        else:
            val = rng.uniform(0.0, 100.0, size=shape)
            data_vars[var] = (
                ("time", "latitude", "longitude"),
                val.astype(np.float32),
                {"units": "dimensionless", "long_name": var},
            )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time_coords,
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
        attrs={
            "title": "SpatioAI Synthetic Meteorological Dataset",
            "source": "SpatioAI Synthetic Data Generator",
            "region": "India and Bay of Bengal",
            "conventions": "CF-1.8",
        },
    )

    return ds


def inject_synthetic_extreme_event(
    ds: xr.Dataset,
    target_time: Optional[str] = None,
    center_lat: float = 20.0,
    center_lon: float = 87.0,
    radius_deg: float = 1.5,
    peak_intensity: float = 140.0,
    variable: str = "precipitation",
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Inject a localized 2D Gaussian extreme weather anomaly pattern into an existing dataset.

    Args:
        ds: Input xarray Dataset.
        target_time: Timestamp string to inject the event into (defaults to middle timestamp).
        center_lat: Center latitude of the synthetic storm.
        center_lon: Center longitude of the synthetic storm.
        radius_deg: Spatial spread (standard deviation) of the storm in degrees.
        peak_intensity: Maximum additive anomaly value at the center.
        variable: Target variable to inject into (default: 'precipitation').

    Returns:
        Tuple[xr.Dataset, Dict[str, Any]]: (Modified dataset with injected event, Event metadata dict)
    """
    ds_out = ds.copy(deep=True)

    if target_time is None:
        mid_idx = len(ds_out.time) // 2
        target_dt = pd.to_datetime(ds_out.time.values[mid_idx])
    else:
        target_dt = pd.to_datetime(target_time)

    # Find closest matching timestamp
    time_diffs = np.abs(pd.to_datetime(ds_out.time.values) - target_dt)
    min_diff_idx = int(np.argmin(time_diffs))
    target_time_val = ds_out.time.values[min_diff_idx]

    lats = ds_out["latitude"].values
    lons = ds_out["longitude"].values

    # 2D coordinate grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 2D Gaussian storm distribution
    dist_sq = (lat_grid - center_lat) ** 2 + (lon_grid - center_lon) ** 2
    gaussian_storm = peak_intensity * np.exp(-dist_sq / (2.0 * (radius_deg ** 2)))

    # Only add where anomaly > 5% of peak (to maintain a clear localized cluster)
    storm_mask = gaussian_storm > (0.05 * peak_intensity)
    gaussian_storm[~storm_mask] = 0.0

    # Add to target variable at target timestamp
    time_slice = ds_out[variable].sel(time=target_time_val).values + gaussian_storm.astype(np.float32)
    ds_out[variable].loc[dict(time=target_time_val)] = time_slice

    # Calculate injected event ground truth characteristics
    active_lats = lat_grid[storm_mask]
    active_lons = lon_grid[storm_mask]
    active_vals = gaussian_storm[storm_mask]

    event_meta = {
        "target_time": str(target_time),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_deg": radius_deg,
        "peak_intensity": float(peak_intensity),
        "expected_min_lat": float(np.min(active_lats)),
        "expected_max_lat": float(np.max(active_lats)),
        "expected_min_lon": float(np.min(active_lons)),
        "expected_max_lon": float(np.max(active_lons)),
        "expected_cell_count": int(np.sum(storm_mask)),
    }

    return ds_out, event_meta
