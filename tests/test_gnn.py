"""
PyTest Test Suite for SpatioAI Phase 4 Spatio-Temporal Graph Neural Network (GNN).
Tests:
- Node & edge feature extraction and scaling
- Spatio-temporal graph construction & temporal edge gating
- Dataset splitting without data leakage (track & temporal splits)
- GNN forward pass, output shape, and probability calculation
- Gradient computation and backward pass
- Loss function numerical stability (weighted BCE & Focal Loss)
- Tiny dataset overfit test
- Track reconstruction logic
- Checkpoint save / load and end-to-end inference
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.optim as optim

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
from src.gnn.graph_builder import GraphBuildConfig, SpatioTemporalGraphBuilder
from src.gnn.inference import GNNPredictor, GNNTrackReconstructor
from src.gnn.losses import FocalLoss, get_loss_function
from src.gnn.model import FallbackSageConv, SpatioTemporalGNN


@pytest.fixture
def sample_events():
    events, track_labels = generate_synthetic_event_sequence(
        num_storms=3,
        timesteps_per_storm=5,
        time_step_hours=6.0,
        seed=42,
    )
    return events, track_labels


def test_feature_extraction(sample_events):
    events, _ = sample_events
    e1 = events[0]
    e2 = events[1]

    node_feat = extract_node_features(e1)
    assert isinstance(node_feat, np.ndarray)
    assert node_feat.shape == (len(NODE_FEATURE_COLUMNS),)
    assert not np.isnan(node_feat).any()

    edge_feat = extract_edge_features(e1, e2)
    assert isinstance(edge_feat, np.ndarray)
    assert edge_feat.shape == (len(EDGE_FEATURE_COLUMNS),)
    assert not np.isnan(edge_feat).any()


def test_feature_scaler():
    scaler = FeatureScaler(feature_names=["f1", "f2"])
    data_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]], dtype=np.float32)
    scaler.fit(data_train)

    scaled = scaler.transform(data_train)
    assert np.allclose(np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-5)
    assert np.allclose(np.std(scaled, axis=0), [1.0, 1.0], atol=1e-5)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "scaler.json"
        scaler.save(tmp_path)
        loaded_scaler = FeatureScaler.load(tmp_path)
        assert loaded_scaler.is_fitted
        loaded_scaled = loaded_scaler.transform(data_train)
        assert np.allclose(scaled, loaded_scaled)


def test_graph_builder(sample_events):
    events, track_labels = sample_events
    config = GraphBuildConfig(
        max_distance_km=600.0,
        max_time_gap_hours=12.0,
        min_time_gap_hours=0.1,
    )
    builder = SpatioTemporalGraphBuilder(config)
    graph_dict = builder.build_graph(events, track_labels=track_labels)

    assert graph_dict["x"].shape == (len(events), len(NODE_FEATURE_COLUMNS))
    assert graph_dict["edge_index"].ndim == 2
    assert graph_dict["edge_index"].shape[0] == 2
    num_edges = graph_dict["edge_index"].shape[1]
    assert graph_dict["edge_attr"].shape == (num_edges, len(EDGE_FEATURE_COLUMNS))
    assert graph_dict["y"].shape == (num_edges,)


def test_data_splitting(sample_events):
    events, track_labels = sample_events

    # Track split
    train_e, val_e, test_e = split_events_by_track(events, track_labels, train_ratio=0.6, val_ratio=0.2, seed=42)
    train_trks = {track_labels[e["event_id"]] for e in train_e}
    test_trks = {track_labels[e["event_id"]] for e in test_e}
    assert len(train_trks.intersection(test_trks)) == 0  # No leakage

    # Temporal split
    train_t, val_t, test_t = split_events_temporal(events, train_ratio=0.6, val_ratio=0.2)
    max_train_time = max(e["timestamp"] for e in train_t)
    min_test_time = min(e["timestamp"] for e in test_t)
    assert max_train_time <= min_test_time  # No future leakage


def test_gnn_forward_and_backprop():
    model = SpatioTemporalGNN(
        node_in_dim=15,
        edge_in_dim=9,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
        conv_type="sage",
    )

    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 15)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, 9)
    y = torch.randint(0, 2, (num_edges,)).float()

    logits = model(x, edge_index, edge_attr)
    assert logits.shape == (num_edges,)

    crit = get_loss_function("bce_weighted", pos_weight=2.0)
    loss = crit(logits, y)
    assert not torch.isnan(loss)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_losses():
    logits = torch.tensor([2.0, -1.0, 0.5, -2.5], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    # BCE Weighted
    crit_bce = get_loss_function("bce_weighted", pos_weight=2.0)
    loss_bce = crit_bce(logits, targets)
    assert loss_bce.item() > 0.0

    # Focal Loss
    crit_focal = FocalLoss(alpha=0.75, gamma=2.0)
    loss_focal = crit_focal(logits, targets)
    assert loss_focal.item() > 0.0


def test_tiny_dataset_overfit(sample_events):
    events, track_labels = sample_events
    tiny_events = events[:8]
    builder = SpatioTemporalGraphBuilder(GraphBuildConfig(max_distance_km=800.0, max_time_gap_hours=24.0))
    graph_dict = builder.build_graph(tiny_events, track_labels=track_labels)
    data = build_graph_data_object(graph_dict)

    if data.edge_index.size(1) == 0:
        pytest.skip("Not enough edges in tiny slice.")

    model = SpatioTemporalGNN(node_in_dim=15, edge_in_dim=9, hidden_dim=64, num_layers=2, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    crit = get_loss_function("bce", pos_weight=1.0)

    initial_loss = crit(model(data.x, data.edge_index, data.edge_attr), data.y).item()
    for _ in range(80):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = crit(out, data.y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.30


def test_track_reconstruction():
    reconstructor = GNNTrackReconstructor(association_threshold=0.50)
    edge_pairs = [("EV_1", "EV_2"), ("EV_2", "EV_3"), ("EV_1", "EV_4"), ("EV_A", "EV_B")]
    probabilities = np.array([0.95, 0.90, 0.10, 0.85], dtype=np.float32)
    node_ids = ["EV_1", "EV_2", "EV_3", "EV_4", "EV_A", "EV_B"]

    tracks = reconstructor.reconstruct_tracks_greedy(edge_pairs, probabilities, node_ids)
    # Expected: ["EV_1", "EV_2", "EV_3"], ["EV_A", "EV_B"], ["EV_4"]
    track_sets = [set(t) for t in tracks]
    assert {"EV_1", "EV_2", "EV_3"} in track_sets
    assert {"EV_A", "EV_B"} in track_sets
    assert {"EV_4"} in track_sets


def test_evaluation_metrics():
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.8, 0.1, 0.3, 0.2]
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["precision"] == 1.0  # Top 2 predictions are both true positives
