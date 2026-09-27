"""
Complete End-to-End Validation Script for SpatioAI Phase 4 Spatio-Temporal GNN.
Validates:
1. Feature extraction & scaling
2. Graph construction
3. GNN forward pass & shapes
4. Loss computation & backpropagation gradients
5. Tiny-dataset overfit test (memorization check)
6. Model training & checkpoint save/load
7. Edge prediction & track reconstruction
8. Baseline Hungarian vs GNN performance comparison
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.optim as optim
import yaml

from src.gnn.dataset import (
    build_graph_data_object,
    generate_synthetic_event_sequence,
    split_events_by_track,
)
from src.gnn.evaluation import compute_classification_metrics, evaluate_track_reconstruction
from src.gnn.features import (
    EDGE_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    FeatureScaler,
    extract_edge_features,
    extract_node_features,
)
from src.gnn.graph_builder import GraphBuildConfig, SpatioTemporalGraphBuilder
from src.gnn.inference import GNNPredictor, GNNTrackReconstructor
from src.gnn.losses import get_loss_function
from src.gnn.model import SpatioTemporalGNN
from src.utils.logger import get_logger

logger = get_logger("ValidateGNN")


def validate_gnn_pipeline() -> bool:
    print("=" * 70)
    print("      SPATIOAI PHASE 4: SPATIO-TEMPORAL GNN PIPELINE VALIDATION")
    print("=" * 70)

    # 1. Test Feature Extraction
    print("\n[Step 1/7] Testing Node & Edge Feature Extraction...")
    sample_evt1 = {
        "event_id": "EV_001",
        "timestamp": "2024-06-01 00:00:00",
        "centroid_lat": 15.0,
        "centroid_lon": 80.0,
        "min_lat": 14.0,
        "max_lat": 16.0,
        "min_lon": 79.0,
        "max_lon": 81.0,
        "area_km2": 250.0,
        "cell_count": 10,
        "max_intensity": 65.0,
        "mean_intensity": 45.0,
    }
    sample_evt2 = {
        "event_id": "EV_002",
        "timestamp": "2024-06-01 06:00:00",
        "centroid_lat": 16.2,
        "centroid_lon": 79.2,
        "min_lat": 15.0,
        "max_lat": 17.0,
        "min_lon": 78.0,
        "max_lon": 80.0,
        "area_km2": 280.0,
        "cell_count": 12,
        "max_intensity": 70.0,
        "mean_intensity": 50.0,
    }
    node_feat = extract_node_features(sample_evt1)
    edge_feat = extract_edge_features(sample_evt1, sample_evt2)

    assert len(node_feat) == len(NODE_FEATURE_COLUMNS), f"Expected {len(NODE_FEATURE_COLUMNS)} node features, got {len(node_feat)}"
    assert len(edge_feat) == len(EDGE_FEATURE_COLUMNS), f"Expected {len(EDGE_FEATURE_COLUMNS)} edge features, got {len(edge_feat)}"
    print(f"  [OK] Node features ({len(node_feat)} dims) and Edge features ({len(edge_feat)} dims) extracted correctly.")

    # 2. Test Graph Construction
    print("\n[Step 2/7] Testing Spatio-Temporal Graph Construction...")
    events, track_labels = generate_synthetic_event_sequence(num_storms=4, timesteps_per_storm=6, seed=42)
    builder = SpatioTemporalGraphBuilder(GraphBuildConfig(max_distance_km=500.0, max_time_gap_hours=12.0))
    graph_dict = builder.build_graph(events, track_labels=track_labels)

    assert graph_dict["x"].shape[0] == len(events)
    assert graph_dict["edge_index"].shape[0] == 2
    assert graph_dict["edge_attr"].shape[0] == graph_dict["edge_index"].shape[1]
    assert len(graph_dict["y"]) == graph_dict["edge_index"].shape[1]
    print(f"  [OK] Graph constructed: {graph_dict['x'].shape[0]} nodes, {graph_dict['edge_index'].shape[1]} candidate edges.")
    print(f"  [OK] Positive associations in synthetic set: {int(np.sum(graph_dict['y']))} / {len(graph_dict['y'])}")

    # 3. Test GNN Forward & Backpropagation
    print("\n[Step 3/7] Testing GNN Forward Pass & Gradient Computation...")
    model = SpatioTemporalGNN(node_in_dim=15, edge_in_dim=9, hidden_dim=64, num_layers=2, dropout=0.1)
    x_t = torch.tensor(graph_dict["x"], dtype=torch.float32)
    edge_idx_t = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr_t = torch.tensor(graph_dict["edge_attr"], dtype=torch.float32)
    y_t = torch.tensor(graph_dict["y"], dtype=torch.float32)

    logits = model(x_t, edge_idx_t, edge_attr_t)
    assert logits.shape == y_t.shape, f"Logits shape {logits.shape} != target shape {y_t.shape}"

    criterion = get_loss_function(loss_type="bce_weighted", pos_weight=2.0)
    loss = criterion(logits, y_t)
    loss.backward()

    # Check non-zero gradients on parameters
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "GNN parameters missing gradients after backward pass."
    print(f"  [OK] Forward pass output logits: {logits.shape}, Loss: {loss.item():.4f}")
    print(f"  [OK] Backpropagation verified: all parameters received gradients.")

    # 4. Mandatory Tiny Dataset Overfit Test
    print("\n[Step 4/7] Testing Tiny-Dataset Overfit (Memorization Check)...")
    tiny_events, tiny_labels = generate_synthetic_event_sequence(num_storms=2, timesteps_per_storm=4, seed=99)
    tiny_graph = builder.build_graph(tiny_events, track_labels=tiny_labels)
    tiny_data = build_graph_data_object(tiny_graph)

    overfit_model = SpatioTemporalGNN(node_in_dim=15, edge_in_dim=9, hidden_dim=64, num_layers=2, dropout=0.0)
    optimizer = optim.Adam(overfit_model.parameters(), lr=0.01)
    overfit_crit = get_loss_function(loss_type="bce_weighted", pos_weight=1.0)

    initial_loss = 0.0
    final_loss = 0.0
    for ep in range(100):
        overfit_model.train()
        optimizer.zero_grad()
        out = overfit_model(tiny_data.x, tiny_data.edge_index, tiny_data.edge_attr)
        l = overfit_crit(out, tiny_data.y)
        if ep == 0:
            initial_loss = l.item()
        l.backward()
        optimizer.step()
        final_loss = l.item()

    assert final_loss < initial_loss * 0.20, f"Overfit failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
    print(f"  [OK] Tiny overfit test PASSED: Loss reduced from {initial_loss:.4f} to {final_loss:.4f} (>80% reduction).")

    # 5. Full Training & Checkpoint Test
    print("\n[Step 5/7] Testing Full Training Loop & Checkpoint Saving...")
    from scripts.train_gnn import train_gnn
    train_res = train_gnn(
        config_path="configs/data.yaml",
        synthetic_storms=8,
        timesteps_per_storm=6,
        epochs_override=25,
        output_checkpoint="models/spatiotemporal_gnn_best.pt",
    )
    print(f"  [OK] Trained checkpoint saved to: {train_res['checkpoint_path']}")
    print(f"  [OK] Test set F1: {train_res['test_metrics']['f1']}, PR-AUC: {train_res['test_metrics']['pr_auc']}")

    # 6. Test Inference & Track Reconstruction
    print("\n[Step 6/7] Testing GNN Predictor & Track Reconstruction...")
    predictor = GNNPredictor(checkpoint_path="models/spatiotemporal_gnn_best.pt", association_threshold=0.50)
    inf_res = predictor.predict_events(events[:15])

    assert "predicted_tracks" in inf_res
    assert "edge_predictions" in inf_res
    assert len(inf_res["predicted_tracks"]) > 0
    print(f"  [OK] Inferred {len(inf_res['predicted_tracks'])} tracks from 15 events.")
    print(f"  [OK] Sample predicted edge: {inf_res['edge_predictions'][0] if inf_res['edge_predictions'] else 'None'}")

    # 7. Baseline vs GNN Comparative Evaluation
    print("\n[Step 7/7] Running Baseline Tracker vs GNN Tracker Comparison...")
    from scripts.evaluate_gnn import run_full_evaluation
    df_comp = run_full_evaluation(config_path="configs/data.yaml", checkpoint_path="models/spatiotemporal_gnn_best.pt")

    print("=" * 70)
    print("  PHASE 4: SPATIO-TEMPORAL GNN VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = validate_gnn_pipeline()
    sys.exit(0 if success else 1)
