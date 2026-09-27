"""
Extreme weather event and anomaly threshold detection for SpatioAI.
"""

from typing import Any, Dict, Optional, Union
import xarray as xr
from src.climatology.anomaly import AnomalyCalculator
from src.climatology.baseline import ClimatologyBaseline
from src.utils.logger import get_logger

logger = get_logger("ExtremeDetection")


class ExtremeDetector:
    """
    Identifies candidate extreme weather anomaly regions and produces binary extreme masks.
    """

    def __init__(
        self,
        baseline: Optional[ClimatologyBaseline] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            baseline: Fitted ClimatologyBaseline instance (required for percentile/standardized detection).
            config: Optional configuration dictionary.
        """
        self.baseline = baseline
        self.config = config or {}

    def detect_by_percentile(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        percentile_stat: str = "p95",
        min_absolute_val: Optional[float] = None,
    ) -> xr.DataArray:
        """
        Detect extreme grid cells where actual values exceed a climatological percentile baseline.

        Args:
            ds: Input weather dataset.
            variable: Variable to check.
            percentile_stat: Baseline percentile statistic (e.g., 'p90', 'p95', 'p99').
            min_absolute_val: Optional minimum physical floor (e.g., min 5 mm for rainfall to prevent dry-pixel triggers).

        Returns:
            xr.DataArray: Binary mask (1 for candidate extreme, 0 for normal).
        """
        if self.baseline is None:
            raise ValueError("ClimatologyBaseline is required for percentile-based extreme detection.")

        actual = ds[variable]
        pct_field = self.baseline.align_to_dataset(ds, stat_name=percentile_stat, variable=variable)

        mask = actual > pct_field

        if min_absolute_val is not None:
            mask = mask & (actual >= min_absolute_val)

        mask_da = mask.astype(int)
        mask_da.name = f"{variable}_extreme_mask"
        mask_da.attrs = {
            "detection_method": "percentile_exceedance",
            "percentile_level": percentile_stat,
            "min_absolute_floor": min_absolute_val if min_absolute_val is not None else "none",
            "description": f"Binary candidate extreme mask ({percentile_stat} exceedance)",
        }
        return mask_da

    def detect_by_standardized_anomaly(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        z_threshold: float = 2.0,
        min_absolute_val: Optional[float] = None,
    ) -> xr.DataArray:
        """
        Detect extreme grid cells where standardized anomaly z >= z_threshold.

        Args:
            ds: Input weather dataset.
            variable: Variable to check.
            z_threshold: Minimum z-score threshold (e.g., 2.0).
            min_absolute_val: Optional physical minimum threshold.

        Returns:
            xr.DataArray: Binary extreme mask.
        """
        if self.baseline is None:
            raise ValueError("ClimatologyBaseline is required for standardized anomaly detection.")

        calculator = AnomalyCalculator(self.baseline)
        z_score = calculator.calculate(ds, variable=variable, method="standardized")

        mask = z_score >= z_threshold

        if min_absolute_val is not None:
            mask = mask & (ds[variable] >= min_absolute_val)

        mask_da = mask.astype(int)
        mask_da.name = f"{variable}_extreme_mask"
        mask_da.attrs = {
            "detection_method": "standardized_z_score",
            "z_threshold": z_threshold,
            "min_absolute_floor": min_absolute_val if min_absolute_val is not None else "none",
            "description": f"Binary candidate extreme mask (z >= {z_threshold})",
        }
        return mask_da

    def detect_by_absolute_threshold(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        threshold: float = 50.0,
    ) -> xr.DataArray:
        """
        Detect extreme grid cells exceeding a fixed physical threshold (e.g., heavy rain > 50 mm).

        Args:
            ds: Input weather dataset.
            variable: Variable name.
            threshold: Value threshold in standard units.

        Returns:
            xr.DataArray: Binary extreme mask.
        """
        actual = ds[variable]
        mask = actual >= threshold

        mask_da = mask.astype(int)
        mask_da.name = f"{variable}_extreme_mask"
        mask_da.attrs = {
            "detection_method": "absolute_threshold",
            "threshold_value": threshold,
            "description": f"Binary candidate extreme mask (value >= {threshold})",
        }
        return mask_da

    def detect(
        self,
        ds: xr.Dataset,
        variable: str = "precipitation",
        method: Optional[str] = None,
    ) -> xr.DataArray:
        """
        Unified detection using configuration or passed method.
        """
        det_method = method or self.config.get("detection_method", "percentile")

        if det_method == "percentile":
            pct_level = self.config.get("percentile_level", "p95")
            min_floor = self.config.get("absolute_min_val", 10.0)
            return self.detect_by_percentile(ds, variable=variable, percentile_stat=pct_level, min_absolute_val=min_floor)

        elif det_method == "standardized":
            z_min = float(self.config.get("standardized_z_min", 2.0))
            min_floor = self.config.get("absolute_min_val", 10.0)
            return self.detect_by_standardized_anomaly(ds, variable=variable, z_threshold=z_min, min_absolute_val=min_floor)

        elif det_method == "absolute":
            thresh = float(self.config.get("absolute_min_val", 50.0))
            return self.detect_by_absolute_threshold(ds, variable=variable, threshold=thresh)

        else:
            raise ValueError(f"Unknown detection method: {det_method}")
