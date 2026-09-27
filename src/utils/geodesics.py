"""
Geodesic distance, bearing, velocity, and bounding box geometric calculations for SpatioAI.
"""

from typing import Any, Tuple, Union
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute great-circle distance between two geographic points on Earth using Haversine formula.

    Args:
        lat1, lon1: Coordinates of first point(s) in decimal degrees.
        lat2, lon2: Coordinates of second point(s) in decimal degrees.

    Returns:
        Great-circle distance in kilometers.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    # Clip to [0, 1] to prevent floating point domain errors in arcsin
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    dist_km = EARTH_RADIUS_KM * c

    if isinstance(dist_km, np.ndarray) and dist_km.ndim == 0:
        return float(dist_km)
    return dist_km


def calculate_bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate initial compass bearing (azimuth) from point 1 to point 2 in degrees (0 to 360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    theta_rad = np.arctan2(y, x)
    theta_deg = (np.degrees(theta_rad) + 360.0) % 360.0
    return float(theta_deg)


def calculate_speed_kmh(
    distance_km: float,
    time1: Any,
    time2: Any,
) -> float:
    """
    Calculate average speed in km/h between two timestamps.

    Args:
        distance_km: Displacement in kilometers.
        time1: Starting timestamp (str, datetime, or Timestamp).
        time2: Ending timestamp.

    Returns:
        Speed in km/h. Returns 0.0 if time delta is <= 0.
    """
    t1 = pd.to_datetime(time1)
    t2 = pd.to_datetime(time2)
    delta_hours = (t2 - t1).total_seconds() / 3600.0

    if delta_hours <= 0.0:
        return 0.0

    speed = distance_km / delta_hours
    return float(speed)


def calculate_bbox_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """
    Calculate Intersection over Union (IoU) of two geographic bounding boxes.

    Args:
        bbox1: Tuple of (min_lat, max_lat, min_lon, max_lon).
        bbox2: Tuple of (min_lat, max_lat, min_lon, max_lon).

    Returns:
        IoU value in range [0.0, 1.0].
    """
    min_lat1, max_lat1, min_lon1, max_lon1 = bbox1
    min_lat2, max_lat2, min_lon2, max_lon2 = bbox2

    inter_min_lat = max(min_lat1, min_lat2)
    inter_max_lat = min(max_lat1, max_lat2)
    inter_min_lon = max(min_lon1, min_lon2)
    inter_max_lon = min(max_lon1, max_lon2)

    inter_lat = max(0.0, inter_max_lat - inter_min_lat)
    inter_lon = max(0.0, inter_max_lon - inter_min_lon)
    intersection = inter_lat * inter_lon

    area1 = max(0.0, max_lat1 - min_lat1) * max(0.0, max_lon1 - min_lon1)
    area2 = max(0.0, max_lat2 - min_lat2) * max(0.0, max_lon2 - min_lon2)

    union = area1 + area2 - intersection

    if union <= 1e-9:
        return 0.0

    iou = intersection / union
    return float(np.clip(iou, 0.0, 1.0))
