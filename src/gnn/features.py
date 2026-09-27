"""
Feature engineering, extraction, and reusable scaling for SpatioAI Spatio-Temporal GNN (Phase 4).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from src.utils.geodesics import (
    calculate_bearing_deg,
    calculate_bbox_iou,
    calculate_speed_kmh,
    haversine_distance_km,
)
from src.utils.logger import get_logger

logger = get_logger("GNNFeatures")

# Standard Node Feature Columns
NODE_FEATURE_COLUMNS = [
    "centroid_lat",
    "centroid_lon",
    "area_km2",
    "cell_count",
    "max_intensity",
    "mean_intensity",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "prev_speed_kmh",
    "prev_bearing_sin",
    "prev_bearing_cos",
    "prev_intensity_change",
    "prev_area_change_ratio",
]

# Standard Edge Feature Columns
EDGE_FEATURE_COLUMNS = [
    "distance_km",
    "bearing_sin",
    "bearing_cos",
    "delta_t_hours",
    "speed_kmh",
    "bbox_iou",
    "area_ratio",
    "intensity_diff",
    "max_intensity_ratio",
]


class FeatureScaler:
    """
    Reusable Mean/Std feature scaler for node and edge feature tensors.
    Prevents data leakage by strictly fitting on training splits only.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, features: Union[np.ndarray, List[List[float]]]) -> "FeatureScaler":
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit FeatureScaler on empty feature matrix.")

        self.mean = np.mean(arr, axis=0)
        self.std = np.std(arr, axis=0)
        # Avoid division by zero for invariant features
        self.std[self.std < 1e-6] = 1.0
        self.is_fitted = True
        return self

    def transform(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform().")
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return (arr - self.mean) / self.std

    def fit_transform(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        return self.fit(features).transform(features)

    def inverse_transform(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling inverse_transform().")
        arr = np.asarray(features, dtype=np.float32)
        return (arr * self.std) + self.mean

    def save(self, filepath: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureScaler.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "is_fitted": self.is_fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(feature_names=data.get("feature_names", []))
        scaler.mean = np.array(data["mean"], dtype=np.float32)
        scaler.std = np.array(data["std"], dtype=np.float32)
        scaler.is_fitted = data["is_fitted"]
        return scaler


def extract_node_features(
    event: Dict[str, Any],
    history_events: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Extract a 1D vector of normalized/engineered features for a single event node.
    No future information is leaked.

    Features:
    - centroid_lat, centroid_lon (geographic)
    - area_km2, cell_count (size)
    - max_intensity, mean_intensity (intensity)
    - hour_sin, hour_cos (diurnal cycle)
    - doy_sin, doy_cos (annual cycle)
    - prev_speed_kmh, prev_bearing_sin, prev_bearing_cos (motion context from past)
    - prev_intensity_change, prev_area_change_ratio (evolution context)
    """
    lat = float(event.get("centroid_lat", 0.0))
    lon = float(event.get("centroid_lon", 0.0))
    area = float(event.get("area_km2", 0.0))
    cell_count = float(event.get("cell_count", 1.0))
    max_int = float(event.get("max_intensity", 0.0))
    mean_int = float(event.get("mean_intensity", 0.0))

    # Cyclic temporal encoding
    ts = pd.to_datetime(event["timestamp"])
    hour = ts.hour + ts.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    doy = ts.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    # Derived motion/context features from immediately preceding event (if available)
    prev_speed = 0.0
    prev_bearing_sin = 0.0
    prev_bearing_cos = 1.0
    prev_intensity_change = 0.0
    prev_area_change_ratio = 1.0

    if history_events and len(history_events) > 0:
        prev_evt = history_events[-1]
        prev_lat = float(prev_evt.get("centroid_lat", lat))
        prev_lon = float(prev_evt.get("centroid_lon", lon))
        prev_dist = haversine_distance_km(prev_lat, prev_lon, lat, lon)
        prev_speed = calculate_speed_kmh(prev_dist, prev_evt["timestamp"], event["timestamp"])

        bearing = calculate_bearing_deg(prev_lat, prev_lon, lat, lon)
        bearing_rad = np.radians(bearing)
        prev_bearing_sin = np.sin(bearing_rad)
        prev_bearing_cos = np.cos(bearing_rad)

        prev_intensity_change = max_int - float(prev_evt.get("max_intensity", max_int))
        prev_area = max(float(prev_evt.get("area_km2", area)), 1.0)
        curr_area = max(area, 1.0)
        prev_area_change_ratio = curr_area / prev_area

    feat = [
        lat,
        lon,
        area,
        cell_count,
        max_int,
        mean_int,
        float(hour_sin),
        float(hour_cos),
        float(doy_sin),
        float(doy_cos),
        float(prev_speed),
        float(prev_bearing_sin),
        float(prev_bearing_cos),
        float(prev_intensity_change),
        float(prev_area_change_ratio),
    ]
    return np.array(feat, dtype=np.float32)


def extract_edge_features(
    source_event: Dict[str, Any],
    target_event: Dict[str, Any],
) -> np.ndarray:
    """
    Extract a 1D vector of edge features between source event (t) and target candidate event (t+dt).

    Features:
    - distance_km: Great-circle distance between centroids
    - bearing_sin, bearing_cos: Compass heading from source to target
    - delta_t_hours: Temporal interval in hours (strictly > 0)
    - speed_kmh: Implied propagation velocity
    - bbox_iou: Spatial overlap of bounding boxes
    - area_ratio: max(A1, A2) / min(A1, A2)
    - intensity_diff: target_max_intensity - source_max_intensity
    - max_intensity_ratio: target_max_intensity / source_max_intensity
    """
    lat1, lon1 = float(source_event["centroid_lat"]), float(source_event["centroid_lon"])
    lat2, lon2 = float(target_event["centroid_lat"]), float(target_event["centroid_lon"])

    dist_km = float(haversine_distance_km(lat1, lon1, lat2, lon2))
    bearing = calculate_bearing_deg(lat1, lon1, lat2, lon2)
    bearing_rad = np.radians(bearing)
    bearing_sin = float(np.sin(bearing_rad))
    bearing_cos = float(np.cos(bearing_rad))

    t1 = pd.to_datetime(source_event["timestamp"])
    t2 = pd.to_datetime(target_event["timestamp"])
    dt_hours = max(float((t2 - t1).total_seconds() / 3600.0), 0.0)

    speed_kmh = calculate_speed_kmh(dist_km, source_event["timestamp"], target_event["timestamp"])

    bbox1 = (
        float(source_event.get("min_lat", lat1)),
        float(source_event.get("max_lat", lat1)),
        float(source_event.get("min_lon", lon1)),
        float(source_event.get("max_lon", lon1)),
    )
    bbox2 = (
        float(target_event.get("min_lat", lat2)),
        float(target_event.get("max_lat", lat2)),
        float(target_event.get("min_lon", lon2)),
        float(target_event.get("max_lon", lon2)),
    )
    iou = float(calculate_bbox_iou(bbox1, bbox2))

    area1 = max(float(source_event.get("area_km2", 1.0)), 1.0)
    area2 = max(float(target_event.get("area_km2", 1.0)), 1.0)
    area_ratio = float(max(area1, area2) / min(area1, area2))

    int1 = max(float(source_event.get("max_intensity", 1.0)), 1e-3)
    int2 = max(float(target_event.get("max_intensity", 1.0)), 1e-3)
    intensity_diff = float(int2 - int1)
    intensity_ratio = float(int2 / int1)

    edge_feat = [
        dist_km,
        bearing_sin,
        bearing_cos,
        dt_hours,
        speed_kmh,
        iou,
        area_ratio,
        intensity_diff,
        intensity_ratio,
    ]
    return np.array(edge_feat, dtype=np.float32)
