"""
Data association and candidate event matching module for SpatioAI Phase 3.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.utils.geodesics import (
    calculate_bbox_iou,
    calculate_speed_kmh,
    haversine_distance_km,
)
from src.utils.logger import get_logger

logger = get_logger("EventAssociation")


class EventAssociationConfig:
    """Configuration parameters for candidate matching and association."""

    def __init__(
        self,
        max_centroid_distance_km: float = 500.0,
        max_speed_kmh: float = 120.0,
        max_area_change_ratio: float = 5.0,
        max_missed_steps: int = 1,
        weight_distance: float = 0.50,
        weight_bbox: float = 0.20,
        weight_area: float = 0.15,
        weight_intensity: float = 0.15,
        algorithm: str = "hungarian",
        cost_threshold: float = 1.0,
    ):
        self.max_centroid_distance_km = max_centroid_distance_km
        self.max_speed_kmh = max_speed_kmh
        self.max_area_change_ratio = max_area_change_ratio
        self.max_missed_steps = max_missed_steps
        self.weight_distance = weight_distance
        self.weight_bbox = weight_bbox
        self.weight_area = weight_area
        self.weight_intensity = weight_intensity
        self.algorithm = algorithm
        self.cost_threshold = cost_threshold


class EventCandidateMatcher:
    """
    Computes pairwise association costs and solves assignment between active tracks and newly detected events.
    """

    def __init__(self, config: Optional[Union[EventAssociationConfig, Dict[str, Any]]] = None):
        if isinstance(config, dict):
            self.config = EventAssociationConfig(**config)
        elif config is not None:
            self.config = config
        else:
            self.config = EventAssociationConfig()

    def compute_pair_cost(
        self,
        track_state: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Tuple[float, bool]:
        """
        Compute association cost between an active track's latest state and a candidate event.

        Args:
            track_state: Dict with 'centroid_lat', 'centroid_lon', 'timestamp', 'min_lat', 'max_lat',
                         'min_lon', 'max_lon', 'area_km2', 'max_intensity'.
            event: Dict with same spatial properties for candidate event at timestamp t+dt.

        Returns:
            Tuple[cost, is_valid_candidate]
        """
        lat1, lon1 = track_state["centroid_lat"], track_state["centroid_lon"]
        lat2, lon2 = event["centroid_lat"], event["centroid_lon"]

        # 1. Geographic distance
        dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
        if dist_km > self.config.max_centroid_distance_km:
            return 1e6, False

        # 2. Motion speed gating
        speed_kmh = calculate_speed_kmh(dist_km, track_state["timestamp"], event["timestamp"])
        if speed_kmh > self.config.max_speed_kmh:
            return 1e6, False

        # 3. Area ratio gating
        area1 = max(float(track_state.get("area_km2", 1.0)), 1.0)
        area2 = max(float(event.get("area_km2", 1.0)), 1.0)
        area_ratio = max(area1, area2) / min(area1, area2)
        if area_ratio > self.config.max_area_change_ratio:
            return 1e6, False

        # Normalized distance cost [0, 1]
        dist_cost = dist_km / max(self.config.max_centroid_distance_km, 1e-3)

        # Bounding box IoU cost [0, 1]
        bbox1 = (track_state["min_lat"], track_state["max_lat"], track_state["min_lon"], track_state["max_lon"])
        bbox2 = (event["min_lat"], event["max_lat"], event["min_lon"], event["max_lon"])
        iou = calculate_bbox_iou(bbox1, bbox2)
        bbox_cost = 1.0 - iou

        # Area difference cost [0, 1]
        area_cost = abs(area1 - area2) / max(area1, area2)

        # Intensity difference cost [0, 1]
        int1 = float(track_state.get("max_intensity", 1.0))
        int2 = float(event.get("max_intensity", 1.0))
        intensity_cost = abs(int1 - int2) / max(int1, int2, 1e-3)

        # Multi-factor weighted association cost
        total_cost = (
            self.config.weight_distance * dist_cost
            + self.config.weight_bbox * bbox_cost
            + self.config.weight_area * area_cost
            + self.config.weight_intensity * intensity_cost
        )

        return float(total_cost), True

    def match_events(
        self,
        active_tracks: List[Dict[str, Any]],
        candidate_events: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        Solve multi-object association between active tracks and new events at current timestamp.

        Args:
            active_tracks: List of track state dictionaries.
            candidate_events: List of new event dictionaries.

        Returns:
            Tuple[matches, unmatched_track_indices, unmatched_event_indices]
            where matches is a list of (track_idx, event_idx, cost).
        """
        n_tracks = len(active_tracks)
        n_events = len(candidate_events)

        if n_tracks == 0:
            return [], [], list(range(n_events))
        if n_events == 0:
            return [], list(range(n_tracks)), []

        # Construct cost matrix
        cost_matrix = np.full((n_tracks, n_events), 1e6, dtype=np.float64)
        valid_matrix = np.zeros((n_tracks, n_events), dtype=bool)

        for i, track_state in enumerate(active_tracks):
            for j, event in enumerate(candidate_events):
                cost, is_valid = self.compute_pair_cost(track_state, event)
                if is_valid and cost <= self.config.cost_threshold:
                    cost_matrix[i, j] = cost
                    valid_matrix[i, j] = True

        matches: List[Tuple[int, int, float]] = []
        matched_tracks = set()
        matched_events = set()

        if self.config.algorithm == "hungarian":
            # Global optimal assignment via Munkres / Hungarian
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if valid_matrix[r, c] and cost_matrix[r, c] <= self.config.cost_threshold:
                    matches.append((int(r), int(c), float(cost_matrix[r, c])))
                    matched_tracks.add(int(r))
                    matched_events.add(int(c))
        else:
            # Greedy lowest-cost matching fallback
            flat_indices = np.argsort(cost_matrix, axis=None)
            for flat_idx in flat_indices:
                r, c = np.unravel_index(flat_idx, cost_matrix.shape)
                if cost_matrix[r, c] > self.config.cost_threshold or not valid_matrix[r, c]:
                    break
                if r not in matched_tracks and c not in matched_events:
                    matches.append((int(r), int(c), float(cost_matrix[r, c])))
                    matched_tracks.add(int(r))
                    matched_events.add(int(c))

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_events = [j for j in range(n_events) if j not in matched_events]

        return matches, unmatched_tracks, unmatched_events
