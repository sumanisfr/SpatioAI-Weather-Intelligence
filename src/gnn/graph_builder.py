"""
Spatio-Temporal Event Graph Construction for SpatioAI Phase 4.
Builds PyTorch Geometric-ready graph structures connecting meteorological event observations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from src.gnn.features import (
    EDGE_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    FeatureScaler,
    extract_edge_features,
    extract_node_features,
)
from src.utils.geodesics import calculate_speed_kmh, haversine_distance_km
from src.utils.logger import get_logger

logger = get_logger("GraphBuilder")


@dataclass
class GraphBuildConfig:
    """Configuration for spatio-temporal event graph builder."""

    max_distance_km: float = 500.0
    max_time_gap_hours: float = 12.0
    min_time_gap_hours: float = 0.1  # Only allow directed edges forward in time
    max_speed_kmh: float = 150.0
    max_area_change_ratio: float = 6.0
    k_nearest_neighbors: Optional[int] = 5  # Top k spatial candidate edges per time gap


class SpatioTemporalGraphBuilder:
    """
    Constructs graph nodes and candidate directed temporal edges from sequences of weather events.
    """

    def __init__(self, config: Optional[Union[GraphBuildConfig, Dict[str, Any]]] = None):
        if isinstance(config, dict):
            self.config = GraphBuildConfig(**config)
        elif config is not None:
            self.config = config
        else:
            self.config = GraphBuildConfig()

    def build_graph(
        self,
        events: List[Dict[str, Any]],
        track_labels: Optional[Dict[str, str]] = None,
        node_scaler: Optional[FeatureScaler] = None,
        edge_scaler: Optional[FeatureScaler] = None,
    ) -> Dict[str, Any]:
        """
        Constructs graph arrays (nodes, edges, attributes, pseudo-labels) from a list of event dicts.

        Args:
            events: List of event dictionaries with centroids, timestamps, areas, intensities.
            track_labels: Optional mapping from event_id -> track_id (from Phase 3 classical tracker).
            node_scaler: Optional fitted FeatureScaler for node features.
            edge_scaler: Optional fitted FeatureScaler for edge features.

        Returns:
            Dictionary containing:
                - 'x': numpy array (N, node_feat_dim)
                - 'edge_index': numpy array (2, E)
                - 'edge_attr': numpy array (E, edge_feat_dim)
                - 'y': numpy array (E,) binary association labels (if track_labels provided)
                - 'node_ids': List[str] event identifiers
                - 'timestamps': List[str] event timestamps
                - 'edge_pairs': List[Tuple[str, str]] (source_event_id, target_event_id)
        """
        if not events:
            return {
                "x": np.empty((0, len(NODE_FEATURE_COLUMNS)), dtype=np.float32),
                "edge_index": np.empty((2, 0), dtype=np.int64),
                "edge_attr": np.empty((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "node_ids": [],
                "timestamps": [],
                "edge_pairs": [],
            }

        # 1. Sort events chronologically
        events_sorted = sorted(events, key=lambda e: pd.to_datetime(e["timestamp"]))
        n_nodes = len(events_sorted)
        node_id_to_idx = {e["event_id"]: i for i, e in enumerate(events_sorted)}
        node_ids = [e["event_id"] for e in events_sorted]
        timestamps = [str(e["timestamp"]) for e in events_sorted]

        # 2. Extract node feature matrix (N, F_node)
        # Group events by track if track_labels available to extract motion history
        track_to_events: Dict[str, List[Dict[str, Any]]] = {}
        node_feats: List[np.ndarray] = []

        for evt in events_sorted:
            evt_id = evt["event_id"]
            history: List[Dict[str, Any]] = []
            if track_labels and evt_id in track_labels:
                trk_id = track_labels[evt_id]
                history = track_to_events.get(trk_id, [])
                track_to_events.setdefault(trk_id, []).append(evt)

            feat = extract_node_features(evt, history_events=history)
            node_feats.append(feat)

        x = np.stack(node_feats, axis=0) if node_feats else np.empty((0, len(NODE_FEATURE_COLUMNS)), dtype=np.float32)

        # 3. Construct candidate spatio-temporal edges
        edge_indices: List[Tuple[int, int]] = []
        edge_attrs: List[np.ndarray] = []
        edge_labels: List[float] = []
        edge_pairs: List[Tuple[str, str]] = []

        # Pre-parse timestamps
        dt_timestamps = [pd.to_datetime(e["timestamp"]) for e in events_sorted]

        for i in range(n_nodes):
            src_evt = events_sorted[i]
            src_t = dt_timestamps[i]
            src_lat = float(src_evt["centroid_lat"])
            src_lon = float(src_evt["centroid_lon"])

            candidates: List[Tuple[float, int, np.ndarray, float, Tuple[str, str]]] = []

            for j in range(i + 1, n_nodes):
                tgt_evt = events_sorted[j]
                tgt_t = dt_timestamps[j]

                # Temporal constraint (strictly forward in time)
                dt_hours = (tgt_t - src_t).total_seconds() / 3600.0
                if dt_hours < self.config.min_time_gap_hours:
                    continue
                if dt_hours > self.config.max_time_gap_hours:
                    # Beyond maximum temporal window
                    continue

                # Spatial distance constraint
                tgt_lat = float(tgt_evt["centroid_lat"])
                tgt_lon = float(tgt_evt["centroid_lon"])
                dist_km = haversine_distance_km(src_lat, src_lon, tgt_lat, tgt_lon)

                if dist_km > self.config.max_distance_km:
                    continue

                # Velocity constraint
                speed_kmh = calculate_speed_kmh(dist_km, src_evt["timestamp"], tgt_evt["timestamp"])
                if speed_kmh > self.config.max_speed_kmh:
                    continue

                # Area ratio constraint
                a1 = max(float(src_evt.get("area_km2", 1.0)), 1.0)
                a2 = max(float(tgt_evt.get("area_km2", 1.0)), 1.0)
                area_ratio = max(a1, a2) / min(a1, a2)
                if area_ratio > self.config.max_area_change_ratio:
                    continue

                # Compute edge features
                edge_feat = extract_edge_features(src_evt, tgt_evt)

                # Label assignment: 1 if both belong to same track, 0 otherwise
                label_val = 0.0
                if track_labels is not None:
                    src_trk = track_labels.get(src_evt["event_id"])
                    tgt_trk = track_labels.get(tgt_evt["event_id"])
                    if src_trk is not None and tgt_trk is not None and src_trk == tgt_trk:
                        label_val = 1.0

                pair = (src_evt["event_id"], tgt_evt["event_id"])
                candidates.append((dist_km, j, edge_feat, label_val, pair))

            # Apply k-nearest neighbors filtering if specified
            if self.config.k_nearest_neighbors is not None and len(candidates) > self.config.k_nearest_neighbors:
                candidates = sorted(candidates, key=lambda c: c[0])[: self.config.k_nearest_neighbors]

            for _, j, edge_feat, label_val, pair in candidates:
                edge_indices.append((i, j))
                edge_attrs.append(edge_feat)
                edge_labels.append(label_val)
                edge_pairs.append(pair)

        if edge_indices:
            edge_index = np.array(edge_indices, dtype=np.int64).T  # Shape: (2, E)
            edge_attr = np.stack(edge_attrs, axis=0)  # Shape: (E, F_edge)
            y = np.array(edge_labels, dtype=np.float32)  # Shape: (E,)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)
            y = np.empty((0,), dtype=np.float32)

        # 4. Optional feature scaling
        if node_scaler is not None and node_scaler.is_fitted and x.shape[0] > 0:
            x = node_scaler.transform(x)

        if edge_scaler is not None and edge_scaler.is_fitted and edge_attr.shape[0] > 0:
            edge_attr = edge_scaler.transform(edge_attr)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
            "node_ids": node_ids,
            "timestamps": timestamps,
            "edge_pairs": edge_pairs,
        }
