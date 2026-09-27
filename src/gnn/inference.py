"""
Inference and Track Reconstruction Engine for SpatioAI GNN (Phase 4).
Loads trained GNN checkpoints, predicts edge association probabilities, and reconstructs predicted tracks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from src.gnn.features import FeatureScaler
from src.gnn.graph_builder import GraphBuildConfig, SpatioTemporalGraphBuilder
from src.gnn.model import SpatioTemporalGNN
from src.utils.logger import get_logger

logger = get_logger("GNNInference")


class GNNTrackReconstructor:
    """
    Reconstructs continuous event tracks from GNN edge association probabilities.
    Supports greedy thresholding and maximum-weight bipartite assignment.
    """

    def __init__(self, association_threshold: float = 0.50):
        self.association_threshold = association_threshold

    def reconstruct_tracks_greedy(
        self,
        edge_pairs: List[Tuple[str, str]],
        probabilities: np.ndarray,
        node_ids: List[str],
        events_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[List[str]]:
        """
        Reconstructs tracks by greedily connecting highest-probability outgoing edges forward in time.
        Enforces one-to-one temporal continuity (no branching).

        Args:
            edge_pairs: List of (source_event_id, target_event_id)
            probabilities: Array of predicted probabilities for each edge
            node_ids: List of all node IDs in the graph
            events_by_id: Optional dictionary of event details

        Returns:
            List of tracks, where each track is a List[event_id] in chronological order.
        """
        if len(edge_pairs) == 0 or len(probabilities) == 0:
            return [[nid] for nid in node_ids]

        # 1. Filter edges above association threshold
        valid_indices = np.where(probabilities >= self.association_threshold)[0]
        if len(valid_indices) == 0:
            return [[nid] for nid in node_ids]

        # Sort valid candidate edges descending by probability
        sorted_indices = valid_indices[np.argsort(-probabilities[valid_indices])]

        parent_of: Dict[str, str] = {}  # child -> parent
        child_of: Dict[str, str] = {}   # parent -> child

        for idx in sorted_indices:
            src, dst = edge_pairs[idx]
            # Ensure src hasn't already sent a child and dst hasn't already chosen a parent
            if src not in child_of and dst not in parent_of:
                # Prevent cyclic self-loops
                curr = dst
                forms_cycle = False
                while curr in child_of:
                    curr = child_of[curr]
                    if curr == src:
                        forms_cycle = True
                        break

                if not forms_cycle:
                    child_of[src] = dst
                    parent_of[dst] = src

        # 2. Build connected track chains starting from track roots
        visited = set()
        tracks: List[List[str]] = []

        for nid in node_ids:
            if nid not in parent_of and nid not in visited:
                # nid is a track start
                track = [nid]
                visited.add(nid)
                curr = nid
                while curr in child_of:
                    curr = child_of[curr]
                    track.append(curr)
                    visited.add(curr)
                tracks.append(track)

        # Include any remaining unvisited nodes
        for nid in node_ids:
            if nid not in visited:
                tracks.append([nid])
                visited.add(nid)

        return tracks


class GNNPredictor:
    """
    End-to-end inference engine for Spatio-Temporal GNN.
    Loads checkpoint, scales features, predicts links, and returns tracks with probabilities.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        association_threshold: float = 0.50,
    ):
        self.device = torch.device(device)
        self.association_threshold = association_threshold
        self.model: Optional[SpatioTemporalGNN] = None
        self.node_scaler: Optional[FeatureScaler] = None
        self.edge_scaler: Optional[FeatureScaler] = None
        self.config: Dict[str, Any] = {}

        self._load_checkpoint(checkpoint_path)
        self.reconstructor = GNNTrackReconstructor(association_threshold=association_threshold)

    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get("config", {})

        model_kwargs = checkpoint.get("model_kwargs", {
            "node_in_dim": 15,
            "edge_in_dim": 9,
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.0,
        })
        self.model = SpatioTemporalGNN(**model_kwargs)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load scalers if present
        if "node_scaler" in checkpoint:
            self.node_scaler = FeatureScaler()
            self.node_scaler.mean = np.array(checkpoint["node_scaler"]["mean"], dtype=np.float32)
            self.node_scaler.std = np.array(checkpoint["node_scaler"]["std"], dtype=np.float32)
            self.node_scaler.is_fitted = True

        if "edge_scaler" in checkpoint:
            self.edge_scaler = FeatureScaler()
            self.edge_scaler.mean = np.array(checkpoint["edge_scaler"]["mean"], dtype=np.float32)
            self.edge_scaler.std = np.array(checkpoint["edge_scaler"]["std"], dtype=np.float32)
            self.edge_scaler.is_fitted = True

    def predict_events(
        self,
        events: List[Dict[str, Any]],
        graph_builder: Optional[SpatioTemporalGraphBuilder] = None,
    ) -> Dict[str, Any]:
        """
        Runs GNN inference over a sequence of weather events.

        Returns:
            Dict containing:
                - 'predicted_tracks': List of reconstructed event chains [ [id1, id2], ... ]
                - 'edge_predictions': List of dicts { 'source': id1, 'target': id2, 'probability': p }
                - 'probabilities': np.ndarray of edge probabilities
        """
        if graph_builder is None:
            graph_builder = SpatioTemporalGraphBuilder()

        graph_dict = graph_builder.build_graph(
            events=events,
            node_scaler=self.node_scaler,
            edge_scaler=self.edge_scaler,
        )

        x = torch.tensor(graph_dict["x"], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float32, device=self.device)

        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            node_ids = graph_dict["node_ids"]
            return {
                "predicted_tracks": [[nid] for nid in node_ids],
                "edge_predictions": [],
                "probabilities": np.empty((0,), dtype=np.float32),
            }

        with torch.no_grad():
            probs = self.model.predict_probabilities(x, edge_index, edge_attr).cpu().numpy()

        edge_pairs = graph_dict["edge_pairs"]
        node_ids = graph_dict["node_ids"]

        predicted_tracks = self.reconstructor.reconstruct_tracks_greedy(
            edge_pairs=edge_pairs,
            probabilities=probs,
            node_ids=node_ids,
        )

        edge_preds = []
        for (src, dst), prob in zip(edge_pairs, probs):
            edge_preds.append({
                "source": src,
                "target": dst,
                "probability": round(float(prob), 4),
                "is_associated": bool(prob >= self.association_threshold),
            })

        return {
            "predicted_tracks": predicted_tracks,
            "edge_predictions": edge_preds,
            "probabilities": probs,
            "edge_pairs": edge_pairs,
            "node_ids": node_ids,
        }
