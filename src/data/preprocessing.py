"""
Data standardization and preprocessing pipelines for SpatioAI.
"""

from typing import Dict, List, Optional
import numpy as np
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("Preprocessing")

# Canonical variable alias dictionary
CANONICAL_VARIABLE_MAP: Dict[str, str] = {
    # Precipitation
    "tp": "precipitation",
    "total_precipitation": "precipitation",
    "precip": "precipitation",
    "rain": "precipitation",
    "pr": "precipitation",
    "prcp": "precipitation",
    # Temperature
    "2t": "temperature",
    "t2m": "temperature",
    "temp": "temperature",
    "temperature_2m": "temperature",
    "tas": "temperature",
    # Wind components
    "10u": "u10",
    "u10m": "u10",
    "u_component_of_wind_10m": "u10",
    "uas": "u10",
    "10v": "v10",
    "v10m": "v10",
    "v_component_of_wind_10m": "v10",
    "vas": "v10",
    # Surface pressure
    "sp": "surface_pressure",
    "ps": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "mslp": "mean_sea_level_pressure",
}

# Coordinate alias map
CANONICAL_COORD_MAP: Dict[str, str] = {
    "lat": "latitude",
    "latitudes": "latitude",
    "lon": "longitude",
    "longitudes": "longitude",
    "long": "longitude",
    "valid_time": "time",
    "datetime": "time",
    "date": "time",
}


def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize coordinate names, ensure monotonic latitude/longitude, and normalize longitudes to [-180, 180] or [0, 360].

    Args:
        ds: Input xarray Dataset.

    Returns:
        xr.Dataset: Dataset with standardized coordinates.
    """
    # 1. Rename coordinates if matching canonical map
    rename_dict = {}
    for coord in ds.coords:
        coord_lower = str(coord).lower()
        if coord_lower in CANONICAL_COORD_MAP and str(coord) != CANONICAL_COORD_MAP[coord_lower]:
            rename_dict[coord] = CANONICAL_COORD_MAP[coord_lower]

    if rename_dict:
        logger.info(f"Renaming coordinates: {rename_dict}")
        ds = ds.rename(rename_dict)

    # 2. Normalize longitudes if necessary (convert [0, 360) with >180 to standard consistent format if mixed)
    if "longitude" in ds.coords:
        lons = ds["longitude"].values
        if np.any(lons > 180.0) and np.all(lons >= 0.0):
            # Dataset uses 0..360 convention; keep or adjust based on region
            # If standardizing to -180..180:
            pass

    # 3. Sort by latitude ascending (or descending consistently) and time ascending
    sort_dims = []
    if "time" in ds.dims:
        sort_dims.append("time")
    if "latitude" in ds.dims:
        sort_dims.append("latitude")
    if "longitude" in ds.dims:
        sort_dims.append("longitude")

    if sort_dims:
        ds = ds.sortby(sort_dims)

    return ds


def standardize_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename dataset data variables to standard internal SpatioAI names.

    Args:
        ds: Input xarray Dataset.

    Returns:
        xr.Dataset: Dataset with canonical variable names.
    """
    rename_dict = {}
    for var in ds.data_vars:
        var_lower = str(var).lower()
        if var_lower in CANONICAL_VARIABLE_MAP and str(var) != CANONICAL_VARIABLE_MAP[var_lower]:
            target_name = CANONICAL_VARIABLE_MAP[var_lower]
            if target_name not in ds.data_vars:
                rename_dict[var] = target_name

    if rename_dict:
        logger.info(f"Standardizing variables: {rename_dict}")
        ds = ds.rename(rename_dict)

    return ds


def convert_units(ds: xr.Dataset, target_units: Optional[Dict[str, str]] = None) -> xr.Dataset:
    """
    Convert meteorological variables to target units with documented conversions.

    Standard conversions:
    - Precipitation: ERA5 provides meters accumulated ('m'). Converted to 'mm' (* 1000).
    - Temperature: If in Celsius and target is 'K', add 273.15. If in Kelvin and target is 'C', subtract 273.15.
    - Surface pressure: If in Pascals ('Pa') and target is 'hPa', divide by 100.

    Args:
        ds: Input xarray Dataset.
        target_units: Dictionary specifying desired unit per variable (e.g., {'precipitation': 'mm', 'temperature': 'K'}).

    Returns:
        xr.Dataset: Dataset with converted units and updated attributes.
    """
    if target_units is None:
        target_units = {
            "precipitation": "mm",
            "temperature": "K",
            "surface_pressure": "hPa",
            "u10": "m/s",
            "v10": "m/s",
        }

    ds_out = ds.copy()

    for var_name in ds_out.data_vars:
        if var_name not in target_units:
            continue

        target_u = target_units[var_name]
        current_u = ds_out[var_name].attrs.get("units", "").strip()

        # Precipitation: m -> mm
        if var_name == "precipitation" and (current_u in ["m", "meters", "metres"] or (current_u == "" and float(ds_out[var_name].max().values) < 0.5)):
            if target_u == "mm":
                logger.info("Converting precipitation from m to mm (multiplying by 1000)")
                ds_out[var_name] = ds_out[var_name] * 1000.0
                ds_out[var_name].attrs["units"] = "mm"
                ds_out[var_name].attrs["conversion_note"] = "Converted from m to mm by SpatioAI"

        # Temperature: degC -> K or K -> degC
        elif var_name == "temperature":
            mean_val = float(ds_out[var_name].mean().values)
            if target_u == "K" and (current_u in ["C", "degC", "celsius", "°C"] or mean_val < 100.0):
                logger.info("Converting temperature from Celsius to Kelvin (+273.15)")
                ds_out[var_name] = ds_out[var_name] + 273.15
                ds_out[var_name].attrs["units"] = "K"
                ds_out[var_name].attrs["conversion_note"] = "Converted from degC to K by SpatioAI"
            elif target_u in ["C", "degC"] and (current_u in ["K", "kelvin"] or mean_val > 150.0):
                logger.info("Converting temperature from Kelvin to Celsius (-273.15)")
                ds_out[var_name] = ds_out[var_name] - 273.15
                ds_out[var_name].attrs["units"] = "degC"
                ds_out[var_name].attrs["conversion_note"] = "Converted from K to degC by SpatioAI"

        # Surface pressure: Pa -> hPa
        elif var_name == "surface_pressure":
            mean_val = float(ds_out[var_name].mean().values)
            if target_u == "hPa" and (current_u in ["Pa", "pascal", "pascals"] or mean_val > 2000.0):
                logger.info("Converting surface_pressure from Pa to hPa (/ 100)")
                ds_out[var_name] = ds_out[var_name] / 100.0
                ds_out[var_name].attrs["units"] = "hPa"
                ds_out[var_name].attrs["conversion_note"] = "Converted from Pa to hPa by SpatioAI"

    return ds_out


def preprocess_dataset(
    ds: xr.Dataset,
    target_units: Optional[Dict[str, str]] = None,
) -> xr.Dataset:
    """
    Complete preprocessing pipeline: coordinate standardization, variable standardization, and unit normalization.

    Args:
        ds: Raw input xarray Dataset.
        target_units: Target units dictionary.

    Returns:
        xr.Dataset: Preprocessed and standardized xarray Dataset.
    """
    ds = standardize_coordinates(ds)
    ds = standardize_variables(ds)
    ds = convert_units(ds, target_units=target_units)
    return ds
