"""
Downscaling PyTorch Dataset and Leakage-Safe Splitting for SpatioAI Phase 5.
Supports paired (Low-Res Input, High-Res Target), baseline bilinear interpolation onto target grid,
and zero-leakage event-track splitting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.downscaling.interpolation import interpolate_field
from src.utils.logger import get_logger

logger = get_logger("DownscalingDataset")


class DownscalingDataset(Dataset):
    """
    PyTorch Dataset for supervised weather downscaling.
    Prepares input tensors (interpolated onto high-res grid) and high-res target tensors.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        interp_method: str = "bilinear",
        normalize: bool = False,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
    ):
        """
        Args:
            samples: List of dictionaries containing 'low_res', 'high_res', 'low_lats', 'low_lons',
                     'high_lats', 'high_lons', and metadata.
            interp_method: Baseline interpolation method to map low-res field to high-res grid ('bilinear', 'nearest').
            normalize: Whether to apply Z-score normalization to inputs and targets.
            norm_mean: Normalization mean.
            norm_std: Normalization standard deviation.
        """
        self.samples = samples
        self.interp_method = interp_method
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = max(norm_std, 1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        low_res = item["low_res"]
        high_res = item["high_res"]
        low_lats = item["low_lats"]
        low_lons = item["low_lons"]
        high_lats = item["high_lats"]
        high_lons = item["high_lons"]

        # Resample low_res field to high_res target grid via interpolation
        interp_baseline = interpolate_field(
            field=low_res,
            source_lat=low_lats,
            source_lon=low_lons,
            target_lat=high_lats,
            target_lon=high_lons,
            method=self.interp_method,
        ).astype(np.float32)

        # Convert to PyTorch tensors with Channel dimension: [1, H, W]
        x_tensor = torch.from_numpy(interp_baseline).unsqueeze(0)
        y_tensor = torch.from_numpy(high_res.astype(np.float32)).unsqueeze(0)
        baseline_tensor = torch.from_numpy(interp_baseline).unsqueeze(0)

        if self.normalize:
            x_tensor = (x_tensor - self.norm_mean) / self.norm_std
            y_tensor = (y_tensor - self.norm_mean) / self.norm_std

        return {
            "x": x_tensor,
            "y": y_tensor,
            "baseline": baseline_tensor,
            "event_id": item.get("event_id", f"EV_{idx:04d}"),
            "track_id": item.get("track_id", f"TRK_{idx:04d}"),
            "timestamp": str(item.get("timestamp", "")),
        }


def split_downscaling_samples(
    samples: List[Dict[str, Any]],
    method: str = "track",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split downscaling paired samples into Train / Val / Test subsets with leakage prevention.

    Args:
        samples: List of sample dictionaries.
        method: 'track' (split by whole weather event track IDs) or 'chronological' (temporal split).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for track shuffling.

    Returns:
        Tuple[train_samples, val_samples, test_samples]
    """
    if len(samples) == 0:
        return [], [], []

    if method == "track":
        # Group by track_id
        track_map: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            tid = s.get("track_id", s.get("event_id", "TRK_DEFAULT"))
            if tid not in track_map:
                track_map[tid] = []
            track_map[tid].append(s)

        unique_tracks = list(track_map.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_tracks)

        n_trks = len(unique_tracks)
        n_train = max(1, int(n_trks * train_ratio))
        n_val = max(1, int(n_trks * val_ratio)) if (n_trks - n_train) >= 2 else max(0, n_trks - n_train)

        train_tracks = set(unique_tracks[:n_train])
        val_tracks = set(unique_tracks[n_train : n_train + n_val])
        test_tracks = set(unique_tracks[n_train + n_val :])

        if len(test_tracks) == 0 and len(val_tracks) > 1:
            test_tracks.add(list(val_tracks).pop())

        train_samples = [s for tid in train_tracks for s in track_map[tid]]
        val_samples = [s for tid in val_tracks for s in track_map[tid]]
        test_samples = [s for tid in test_tracks for s in track_map[tid]]

        logger.info(
            f"Track-based split: {len(unique_tracks)} tracks -> Train: {len(train_tracks)} tracks ({len(train_samples)} samples), "
            f"Val: {len(val_tracks)} tracks ({len(val_samples)} samples), Test: {len(test_tracks)} tracks ({len(test_samples)} samples)"
        )
        return train_samples, val_samples, test_samples

    elif method == "chronological":
        sorted_samples = sorted(samples, key=lambda s: pd.to_datetime(s["timestamp"]))
        n = len(sorted_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_samples = sorted_samples[:n_train]
        val_samples = sorted_samples[n_train : n_train + n_val]
        test_samples = sorted_samples[n_train + n_val :]

        logger.info(
            f"Chronological split: {n} samples -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}"
        )
        return train_samples, val_samples, test_samples

    else:
        raise ValueError(f"Unknown split method: {method}")
