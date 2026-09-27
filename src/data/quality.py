"""
Quality control and dataset validation for meteorological datasets.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("QualityControl")


class QualityControlError(Exception):
    """Exception raised when a dataset fails critical quality control checks."""
    pass


class DatasetValidator:
    """
    Validates meteorological xarray datasets against physical bounds and structural invariants.
    """

    # Physical valid ranges (min, max) for common meteorological variables
    PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
        "precipitation": (0.0, 1000.0),       # mm per time step
        "temperature": (180.0, 340.0),        # Kelvin
        "temperature_celsius": (-90.0, 65.0), # Celsius
        "u10": (-120.0, 120.0),               # m/s
        "v10": (-120.0, 120.0),               # m/s
        "surface_pressure": (500.0, 1100.0),  # hPa
        "surface_pressure_pa": (50000.0, 110000.0), # Pa
        "relative_humidity": (0.0, 100.0),    # %
    }

    def __init__(
        self,
        max_nan_percentage: float = 5.0,
        enforce_monotonic_time: bool = True,
        enforce_coordinate_bounds: bool = True,
    ):
        self.max_nan_percentage = max_nan_percentage
        self.enforce_monotonic_time = enforce_monotonic_time
        self.enforce_coordinate_bounds = enforce_coordinate_bounds

    def validate(
        self,
        ds: xr.Dataset,
        expected_variables: Optional[List[str]] = None,
        raise_on_failure: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validate an xarray Dataset.

        Args:
            ds: xarray.Dataset to validate.
            expected_variables: Optional list of variable names that must be present.
            raise_on_failure: If True, raises QualityControlError on failure.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
        """
        errors: List[str] = []

        if not isinstance(ds, xr.Dataset):
            errors.append(f"Input is not an xarray.Dataset instance, got {type(ds)}")
            if raise_on_failure:
                raise QualityControlError(errors[0])
            return False, errors

        # 1. Coordinate presence check
        req_coords = ["time", "latitude", "longitude"]
        for coord in req_coords:
            if coord not in ds.coords:
                errors.append(f"Missing required coordinate: '{coord}'")

        if errors:
            if raise_on_failure:
                raise QualityControlError("; ".join(errors))
            return False, errors

        # 2. Coordinate monotonicity & validity
        if self.enforce_monotonic_time:
            time_vals = ds["time"].values
            if len(time_vals) > 1 and not np.all(time_vals[:-1] <= time_vals[1:]):
                errors.append("Time coordinate is not monotonically non-decreasing.")

            # Check for duplicate timestamps
            if len(time_vals) != len(np.unique(time_vals)):
                errors.append("Duplicate timestamps detected in time coordinate.")

        lats = ds["latitude"].values
        lons = ds["longitude"].values

        if self.enforce_coordinate_bounds:
            if np.any((lats < -90.0) | (lats > 90.0)):
                errors.append(f"Latitude coordinate has out-of-range values: [{np.min(lats)}, {np.max(lats)}]")
            if np.any((lons < -180.0) | (lons > 360.0)):
                errors.append(f"Longitude coordinate has out-of-range values: [{np.min(lons)}, {np.max(lons)}]")

        # 3. Check expected variables
        if expected_variables:
            for var in expected_variables:
                if var not in ds.data_vars:
                    errors.append(f"Required variable '{var}' missing from dataset data_vars.")

        # 4. Check NaN / missing values and physical bounds per variable
        for var_name, data_array in ds.data_vars.items():
            total_elements = data_array.size
            if total_elements == 0:
                errors.append(f"Variable '{var_name}' is empty.")
                continue

            nan_count = int(data_array.isnull().sum().values)
            nan_pct = (nan_count / total_elements) * 100.0

            if nan_pct > self.max_nan_percentage:
                errors.append(
                    f"Variable '{var_name}' contains {nan_pct:.2f}% missing/NaN values "
                    f"(exceeds allowed {self.max_nan_percentage}% threshold)."
                )

            # Check physical bounds if available
            bounds = self.PHYSICAL_BOUNDS.get(var_name)
            if bounds is not None:
                min_bound, max_bound = bounds
                var_min = float(data_array.min().values)
                var_max = float(data_array.max().values)
                if var_min < min_bound or var_max > max_bound:
                    errors.append(
                        f"Variable '{var_name}' value range [{var_min:.2f}, {var_max:.2f}] "
                        f"violates physical limits [{min_bound}, {max_bound}]."
                    )

        is_valid = len(errors) == 0
        if not is_valid:
            error_msg = f"Dataset validation failed with {len(errors)} error(s):\n- " + "\n- ".join(errors)
            logger.error(error_msg)
            if raise_on_failure:
                raise QualityControlError(error_msg)
        else:
            logger.info("Dataset validation passed successfully.")

        return is_valid, errors


def validate_dataset(
    ds: xr.Dataset,
    expected_variables: Optional[List[str]] = None,
    max_nan_percentage: float = 5.0,
    raise_on_failure: bool = True,
) -> Tuple[bool, List[str]]:
    """Convenience helper function for dataset validation."""
    validator = DatasetValidator(max_nan_percentage=max_nan_percentage)
    return validator.validate(
        ds=ds,
        expected_variables=expected_variables,
        raise_on_failure=raise_on_failure,
    )
