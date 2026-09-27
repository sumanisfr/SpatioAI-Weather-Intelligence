"""
Evaluation metrics and baseline comparison suite for SpatioAI GNN (Phase 4).
Computes classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
and tracking-oriented metrics (Continuity, Association Accuracy, False Association Rate).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils.logger import get_logger

logger = get_logger("GNNEvaluation")


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard binary edge classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted association probabilities in [0, 1].
        threshold: Classification decision boundary threshold.

    Returns:
        Dict of metrics: precision, recall, f1, roc_auc, pr_auc, positive_count, total_count.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "positive_count": 0,
            "total_count": 0,
        }

    y_pred = (y_prob >= threshold).astype(np.int32)
    n_pos = int(np.sum(y_true))
    n_total = len(y_true)

    # If only one class is present in y_true, AUC cannot be defined standardly
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(n_pos / max(n_total, 1))
    else:
        roc_auc = 0.5
        pr_auc = float(n_pos / max(n_total, 1))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "positive_count": n_pos,
        "total_count": n_total,
    }


def evaluate_track_reconstruction(
    predicted_tracks: List[List[str]],
    reference_tracks: List[List[str]],
) -> Dict[str, float]:
    """
    Evaluates tracking-specific metrics comparing reconstructed tracks with reference tracks.

    Metrics:
    - track_continuity: Mean fraction of consecutive step transitions preserved
    - association_accuracy: Jaccard overlap of event clusters
    - false_association_rate: Ratio of spurious transitions created
    - missed_association_rate: Ratio of reference transitions missed
    """
    # Extract consecutive edge transitions from reference tracks
    ref_edges = set()
    for trk in reference_tracks:
        for i in range(len(trk) - 1):
            ref_edges.add((trk[i], trk[i + 1]))

    # Extract consecutive edge transitions from predicted tracks
    pred_edges = set()
    for trk in predicted_tracks:
        for i in range(len(trk) - 1):
            pred_edges.add((trk[i], trk[i + 1]))

    if not ref_edges:
        return {
            "track_continuity": 1.0 if not pred_edges else 0.0,
            "association_accuracy": 1.0 if not pred_edges else 0.0,
            "false_association_rate": 0.0,
            "missed_association_rate": 0.0,
            "ref_edge_count": 0,
            "pred_edge_count": len(pred_edges),
        }

    tp_edges = len(ref_edges.intersection(pred_edges))
    fp_edges = len(pred_edges - ref_edges)
    fn_edges = len(ref_edges - pred_edges)

    continuity = tp_edges / len(ref_edges)
    fa_rate = fp_edges / max(len(pred_edges), 1)
    miss_rate = fn_edges / len(ref_edges)
    accuracy = tp_edges / max(len(ref_edges.union(pred_edges)), 1)

    return {
        "track_continuity": round(float(continuity), 4),
        "association_accuracy": round(float(accuracy), 4),
        "false_association_rate": round(float(fa_rate), 4),
        "missed_association_rate": round(float(miss_rate), 4),
        "ref_edge_count": len(ref_edges),
        "pred_edge_count": len(pred_edges),
    }


def compare_baseline_vs_gnn(
    baseline_metrics: Dict[str, Any],
    gnn_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """
    Creates a factual comparative dataframe comparing Classical Baseline vs GNN.
    """
    keys = [
        ("Association Precision", "precision"),
        ("Association Recall", "recall"),
        ("Association F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Track Continuity", "track_continuity"),
        ("Association Accuracy", "association_accuracy"),
        ("False Association Rate", "false_association_rate"),
        ("Missed Association Rate", "missed_association_rate"),
    ]

    records = []
    for label, key in keys:
        b_val = baseline_metrics.get(key, "N/A")
        g_val = gnn_metrics.get(key, "N/A")
        records.append({
            "Metric": label,
            "Baseline Tracker": b_val,
            "Spatio-Temporal GNN": g_val,
        })

    return pd.DataFrame(records)
