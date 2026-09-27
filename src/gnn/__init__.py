"""
Spatio-Temporal Graph Neural Network (GNN) module for extreme weather event tracking and evolution.
"""

from src.gnn.dataset import (
    EventGraphData,
    WeatherEventGraphDataset,
    build_graph_data_object,
    generate_synthetic_event_sequence,
    split_events_by_track,
    split_events_temporal,
)
from src.gnn.evaluation import (
    compare_baseline_vs_gnn,
    compute_classification_metrics,
    evaluate_track_reconstruction,
)
from src.gnn.features import (
    EDGE_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    FeatureScaler,
    extract_edge_features,
    extract_node_features,
)
from src.gnn.graph_builder import (
    GraphBuildConfig,
    SpatioTemporalGraphBuilder,
)
from src.gnn.inference import (
    GNNPredictor,
    GNNTrackReconstructor,
)
from src.gnn.losses import (
    FocalLoss,
    get_loss_function,
)
from src.gnn.model import (
    SpatioTemporalGNN,
)

__all__ = [
    "NODE_FEATURE_COLUMNS",
    "EDGE_FEATURE_COLUMNS",
    "FeatureScaler",
    "extract_node_features",
    "extract_edge_features",
    "GraphBuildConfig",
    "SpatioTemporalGraphBuilder",
    "SpatioTemporalGNN",
    "FocalLoss",
    "get_loss_function",
    "EventGraphData",
    "WeatherEventGraphDataset",
    "generate_synthetic_event_sequence",
    "split_events_temporal",
    "split_events_by_track",
    "build_graph_data_object",
    "GNNTrackReconstructor",
    "GNNPredictor",
    "compute_classification_metrics",
    "evaluate_track_reconstruction",
    "compare_baseline_vs_gnn",
]
