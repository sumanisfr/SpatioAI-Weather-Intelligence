"""
Spatial connected component segmentation, event feature extraction, and candidate filtering for SpatioAI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label
import xarray as xr
from src.utils.logger import get_logger

logger = get_logger("EventSegmentation")

EARTH_RADIUS_KM = 6371.0088


def compute_grid_cell_areas_km2(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute 2D array of physical grid cell surface areas in km^2 on a sphere.

    Args:
        lats: 1D array of latitude coordinates in degrees.
        lons: 1D array of longitude coordinates in degrees.

    Returns:
        2D numpy array of shape (len(lats), len(lons)) containing cell areas in km^2.
    """
    n_lat = len(lats)
    n_lon = len(lons)

    if n_lat < 2 or n_lon < 2:
        # Fallback for single cell
        return np.ones((n_lat, n_lon), dtype=np.float64) * 100.0

    d_lon_deg = float(np.abs(np.mean(np.diff(lons))))
    d_lat_deg = float(np.abs(np.mean(np.diff(lats))))

    d_lon_rad = np.radians(d_lon_deg)
    d_lat_rad = np.radians(d_lat_deg)

    # Compute latitude cell bounds
    lat_rad = np.radians(lats)
    lat_bounds_low = lat_rad - (d_lat_rad / 2.0)
    lat_bounds_high = lat_rad + (d_lat_rad / 2.0)

    # Area = R^2 * d_lon * |sin(lat_high) - sin(lat_low)|
    lat_area_1d = (EARTH_RADIUS_KM ** 2) * d_lon_rad * np.abs(np.sin(lat_bounds_high) - np.sin(lat_bounds_low))

    # Broadcast to 2D (n_lat, n_lon)
    area_2d = np.repeat(lat_area_1d[:, np.newaxis], n_lon, axis=1)
    return area_2d


class EventSegmenter:
    """
    Segments 2D binary extreme anomaly masks into labeled connected components,
    extracts spatial properties (centroid, bounding box, area in km^2, intensity),
    and filters noise based on configurable thresholds.
    """

    def __init__(
        self,
        connectivity: int = 8,
        minimum_cells: int = 4,
        minimum_area_km2: float = 50.0,
    ):
        """
        Args:
            connectivity: 4 or 8 neighborhood connectivity for 2D components.
            minimum_cells: Minimum connected grid cells to qualify as an event.
            minimum_area_km2: Minimum physical area in km^2.
        """
        self.connectivity = connectivity
        self.minimum_cells = minimum_cells
        self.minimum_area_km2 = minimum_area_km2

        # Generate binary structuring element
        rank = 2
        connectivity_level = 2 if connectivity == 8 else 1
        self.structure = generate_binary_structure(rank, connectivity_level)

    def segment_slice(
        self,
        mask_2d: np.ndarray,
        intensity_2d: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        timestamp: Any,
        cell_areas_2d: Optional[np.ndarray] = None,
        event_prefix: str = "EV",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segment a single 2D spatial slice into labeled event components and extract features.

        Args:
            mask_2d: 2D binary array (1 = extreme, 0 = normal).
            intensity_2d: 2D array of actual physical values (e.g. precipitation in mm).
            lats: 1D latitude coordinates.
            lons: 1D longitude coordinates.
            timestamp: Timestamp of the slice.
            cell_areas_2d: Precomputed cell areas in km^2.
            event_prefix: Identifier prefix.

        Returns:
            Tuple[labeled_mask, event_feature_dicts]
        """
        if cell_areas_2d is None:
            cell_areas_2d = compute_grid_cell_areas_km2(lats, lons)

        labeled_array, num_features = label(mask_2d, structure=self.structure)
        events_list: List[Dict[str, Any]] = []

        ts_str = pd.to_datetime(str(timestamp)).strftime("%Y%m%d_%H%M")

        for feat_id in range(1, num_features + 1):
            cell_mask = (labeled_array == feat_id)
            cell_count = int(np.sum(cell_mask))

            if cell_count < self.minimum_cells:
                labeled_array[cell_mask] = 0
                continue

            event_area = float(np.sum(cell_areas_2d[cell_mask]))
            if event_area < self.minimum_area_km2:
                labeled_array[cell_mask] = 0
                continue

            # Coordinates of cells
            y_indices, x_indices = np.where(cell_mask)
            event_lats = lats[y_indices]
            event_lons = lons[x_indices]
            event_intensities = intensity_2d[cell_mask]

            # Intensity-weighted centroid if intensity sum > 0, else geometric
            int_sum = float(np.sum(event_intensities))
            if int_sum > 1e-6:
                centroid_lat = float(np.sum(event_lats * event_intensities) / int_sum)
                centroid_lon = float(np.sum(event_lons * event_intensities) / int_sum)
            else:
                centroid_lat = float(np.mean(event_lats))
                centroid_lon = float(np.mean(event_lons))

            event_id = f"{event_prefix}_{ts_str}_{feat_id:03d}"

            event_info = {
                "event_id": event_id,
                "timestamp": str(timestamp),
                "cell_count": cell_count,
                "area_km2": round(event_area, 2),
                "centroid_lat": round(centroid_lat, 4),
                "centroid_lon": round(centroid_lon, 4),
                "min_lat": round(float(np.min(event_lats)), 4),
                "max_lat": round(float(np.max(event_lats)), 4),
                "min_lon": round(float(np.min(event_lons)), 4),
                "max_lon": round(float(np.max(event_lons)), 4),
                "max_intensity": round(float(np.max(event_intensities)), 2),
                "mean_intensity": round(float(np.mean(event_intensities)), 2),
            }
            events_list.append(event_info)

        return labeled_array, events_list

    def extract_events(
        self,
        mask_da: xr.DataArray,
        intensity_da: xr.DataArray,
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """
        Segment all timestamps in a 3D (time, latitude, longitude) DataArray.

        Args:
            mask_da: 3D binary extreme mask.
            intensity_da: 3D physical intensity values (e.g. precipitation).

        Returns:
            Tuple[labeled_mask_da, events_dataframe]
        """
        lats = mask_da["latitude"].values
        lons = mask_da["longitude"].values
        times = mask_da["time"].values

        cell_areas = compute_grid_cell_areas_km2(lats, lons)

        labeled_3d = np.zeros_like(mask_da.values, dtype=np.int32)
        all_events: List[Dict[str, Any]] = []

        for t_idx, t_val in enumerate(times):
            mask_2d = mask_da.isel(time=t_idx).values.astype(int)
            int_2d = intensity_da.isel(time=t_idx).values.astype(np.float64)

            labeled_2d, events = self.segment_slice(
                mask_2d=mask_2d,
                intensity_2d=int_2d,
                lats=lats,
                lons=lons,
                timestamp=t_val,
                cell_areas_2d=cell_areas,
            )
            labeled_3d[t_idx] = labeled_2d
            all_events.extend(events)

        labeled_da = xr.DataArray(
            labeled_3d,
            coords=mask_da.coords,
            dims=mask_da.dims,
            name="labeled_extreme_events",
            attrs={"description": "Labeled candidate extreme weather objects"},
        )

        df_events = pd.DataFrame(all_events)
        logger.info(f"Extracted {len(df_events)} candidate extreme weather events across {len(times)} timestamps.")
        return labeled_da, df_events

    @staticmethod
    def save_events_table(
        df_events: pd.DataFrame,
        output_path: Union[str, Path],
        file_format: str = "parquet",
    ) -> Path:
        """
        Save extracted events table to Parquet or CSV.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "parquet" or path.suffix == ".parquet":
            if not str(path).endswith(".parquet"):
                path = path.with_suffix(".parquet")
            try:
                df_events.to_parquet(str(path), index=False)
                logger.info(f"Saved {len(df_events)} events to Parquet at {path}")
            except Exception as e:
                logger.warning(f"Parquet engine error ({e}); falling back to CSV.")
                csv_path = path.with_suffix(".csv")
                df_events.to_csv(str(csv_path), index=False)
                return csv_path
        else:
            if not str(path).endswith(".csv"):
                path = path.with_suffix(".csv")
            df_events.to_csv(str(path), index=False)
            logger.info(f"Saved {len(df_events)} events to CSV at {path}")

        return path
