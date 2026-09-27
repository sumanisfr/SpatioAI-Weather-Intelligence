"""
Graph Dataset, Synthetic Data Generation, and Temporal/Track Splitting for SpatioAI GNN.
Supports PyTorch / PyTorch Geometric Data structures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.gnn.features import FeatureScaler
from src.gnn.graph_builder import GraphBuildConfig, SpatioTemporalGraphBuilder
from src.utils.logger import get_logger

logger = get_logger("GNNDataset")

try:
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class EventGraphData:
    """
    Lightweight container for single event spatio-temporal graph.
    Compatible with standard PyTorch and PyTorch Geometric.
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        node_ids: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
        edge_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.node_ids = node_ids or []
        self.timestamps = timestamps or []
        self.edge_pairs = edge_pairs or []

    def to(self, device: Union[str, torch.device]) -> "EventGraphData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self

    def to_pyg_data(self) -> Any:
        if not PYG_AVAILABLE:
            raise RuntimeError("PyG is not installed.")
        return PyGData(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y,
        )


class WeatherEventGraphDataset(Dataset):
    """
    PyTorch Dataset of Spatio-Temporal Event Graphs.
    """

    def __init__(self, graph_items: List[EventGraphData]):
        self.graph_items = graph_items

    def __len__(self) -> int:
        return len(self.graph_items)

    def __getitem__(self, idx: int) -> EventGraphData:
        return self.graph_items[idx]


def generate_synthetic_event_sequence(
    num_storms: int = 4,
    timesteps_per_storm: int = 6,
    time_step_hours: float = 6.0,
    seed: int = 42,
    noise_level: float = 0.05,
    random_drop_rate: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Generates realistic synthetic multi-storm trajectories with movement, area changes,
    intensity fluctuations, and temporary missed detections.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, str]]: (events_list, event_id_to_track_id)
    """
    rng = np.random.default_rng(seed)
    base_time = pd.to_datetime("2024-06-01 00:00:00")
    events: List[Dict[str, Any]] = []
    track_labels: Dict[str, str] = {}

    # Define base trajectory origins and vectors for different storm systems in India region
    storm_origins = [
        {"track_id": "TRACK_BOB_01", "lat0": 12.0, "lon0": 88.0, "dlat": 1.2, "dlon": -0.8, "base_int": 75.0, "base_area": 350.0},
        {"track_id": "TRACK_AS_01", "lat0": 15.0, "lon0": 68.0, "dlat": 0.9, "dlon": 1.1, "base_int": 60.0, "base_area": 280.0},
        {"track_id": "TRACK_NE_01", "lat0": 24.0, "lon0": 90.0, "dlat": -0.4, "dlon": 0.6, "base_int": 85.0, "base_area": 420.0},
        {"track_id": "TRACK_INLAND_01", "lat0": 18.0, "lon0": 78.0, "dlat": 0.5, "dlon": 0.2, "base_int": 50.0, "base_area": 200.0},
    ]

    for storm_idx in range(num_storms):
        origin = storm_origins[storm_idx % len(storm_origins)]
        track_id = f"{origin['track_id']}_{storm_idx}"

        curr_lat = origin["lat0"] + rng.normal(0, 0.5)
        curr_lon = origin["lon0"] + rng.normal(0, 0.5)
        curr_int = origin["base_int"]
        curr_area = origin["base_area"]

        for t in range(timesteps_per_storm):
            # Simulate occasional missed detection (e.g. cloud obscuration)
            if t > 0 and t < timesteps_per_storm - 1 and rng.random() < random_drop_rate:
                # Skip this timestep for this storm
                curr_lat += origin["dlat"] + rng.normal(0, 0.1)
                curr_lon += origin["dlon"] + rng.normal(0, 0.1)
                continue

            curr_time = base_time + pd.Timedelta(hours=t * time_step_hours + (storm_idx * 12.0))
            ts_str = curr_time.strftime("%Y%m%d_%H%M")
            evt_id = f"EV_{track_id}_{ts_str}"

            # Propagate storm dynamics
            curr_lat += origin["dlat"] + rng.normal(0, noise_level)
            curr_lon += origin["dlon"] + rng.normal(0, noise_level)
            curr_int = max(curr_int + rng.normal(0, 5.0), 20.0)
            curr_area = max(curr_area + rng.normal(0, 25.0), 60.0)

            lat_span = np.sqrt(curr_area) / 111.0 * 0.5
            lon_span = np.sqrt(curr_area) / 111.0 * 0.5

            evt = {
                "event_id": evt_id,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "centroid_lat": round(float(curr_lat), 4),
                "centroid_lon": round(float(curr_lon), 4),
                "min_lat": round(float(curr_lat - lat_span), 4),
                "max_lat": round(float(curr_lat + lat_span), 4),
                "min_lon": round(float(curr_lon - lon_span), 4),
                "max_lon": round(float(curr_lon + lon_span), 4),
                "area_km2": round(float(curr_area), 2),
                "cell_count": int(max(4, curr_area / 20.0)),
                "max_intensity": round(float(curr_int), 2),
                "mean_intensity": round(float(curr_int * 0.7), 2),
            }
            events.append(evt)
            track_labels[evt_id] = track_id

    # Add a few spurious isolated false-alarm events
    for fa_idx in range(num_storms * 2):
        t_fa = rng.integers(0, timesteps_per_storm)
        fa_time = base_time + pd.Timedelta(hours=t_fa * time_step_hours)
        ts_str = fa_time.strftime("%Y%m%d_%H%M")
        fa_id = f"EV_FALSE_ALARM_{ts_str}_{fa_idx:02d}"
        fa_evt = {
            "event_id": fa_id,
            "timestamp": fa_time.strftime("%Y-%m-%d %H:%M:%S"),
            "centroid_lat": round(float(rng.uniform(8.0, 28.0)), 4),
            "centroid_lon": round(float(rng.uniform(68.0, 95.0)), 4),
            "min_lat": 10.0,
            "max_lat": 11.0,
            "min_lon": 70.0,
            "max_lon": 71.0,
            "area_km2": round(float(rng.uniform(50.0, 100.0)), 2),
            "cell_count": int(rng.integers(4, 10)),
            "max_intensity": round(float(rng.uniform(25.0, 40.0)), 2),
            "mean_intensity": round(float(rng.uniform(20.0, 30.0)), 2),
        }
        events.append(fa_evt)
        track_labels[fa_id] = f"FA_TRACK_{fa_idx}"

    return events, track_labels


def split_events_temporal(
    events: List[Dict[str, Any]],
    track_labels: Optional[Dict[str, str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits events temporally to avoid data leakage across train/validation/test splits.
    """
    sorted_events = sorted(events, key=lambda e: pd.to_datetime(e["timestamp"]))
    n = len(sorted_events)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_evts = sorted_events[:n_train]
    val_evts = sorted_events[n_train : n_train + n_val]
    test_evts = sorted_events[n_train + n_val :]

    return train_evts, val_evts, test_evts


def split_events_by_track(
    events: List[Dict[str, Any]],
    track_labels: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits whole event tracks into train/val/test splits.
    Guarantees zero track overlap between splits.
    """
    rng = np.random.default_rng(seed)
    unique_tracks = list(set(track_labels.values()))
    rng.shuffle(unique_tracks)

    n_tracks = len(unique_tracks)
    n_train = max(1, int(n_tracks * train_ratio))
    n_val = max(1, int(n_tracks * val_ratio))

    train_trks = set(unique_tracks[:n_train])
    val_trks = set(unique_tracks[n_train : n_train + n_val])
    test_trks = set(unique_tracks[n_train + n_val :])

    train_evts = [e for e in events if track_labels.get(e["event_id"]) in train_trks]
    val_evts = [e for e in events if track_labels.get(e["event_id"]) in val_trks]
    test_evts = [e for e in events if track_labels.get(e["event_id"]) in test_trks]

    return train_evts, val_evts, test_evts


def build_graph_data_object(
    graph_dict: Dict[str, Any],
) -> EventGraphData:
    """
    Converts graph dictionary output into EventGraphData container with PyTorch Tensors.
    """
    x = torch.tensor(graph_dict["x"], dtype=torch.float32)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float32)
    y = torch.tensor(graph_dict["y"], dtype=torch.float32) if len(graph_dict["y"]) > 0 else None

    return EventGraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_ids=graph_dict.get("node_ids", []),
        timestamps=graph_dict.get("timestamps", []),
        edge_pairs=graph_dict.get("edge_pairs", []),
    )
