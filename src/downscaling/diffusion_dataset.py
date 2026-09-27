"""Dataset adapter for conditional precipitation diffusion downscaling."""

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.downscaling.dataset import split_downscaling_samples
from src.downscaling.diffusion import log1p_transform
from src.downscaling.interpolation import interpolate_field


class DiffusionDownscalingDataset(Dataset):
    """Create transformed target/condition pairs from Phase 5 samples."""

    def __init__(self, samples: List[Dict[str, Any]], interp_method: str = "bilinear", transform: str = "log1p") -> None:
        if transform.lower() != "log1p":
            raise ValueError("Only the 'log1p' precipitation transform is currently supported")
        self.samples = samples
        self.interp_method = interp_method
        self.transform = transform.lower()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        condition = interpolate_field(
            item["low_res"], item["low_lats"], item["low_lons"], item["high_lats"], item["high_lons"], method=self.interp_method
        ).astype(np.float32)
        target = np.asarray(item["high_res"], dtype=np.float32)
        condition_tensor = log1p_transform(torch.from_numpy(condition).unsqueeze(0))
        target_tensor = log1p_transform(torch.from_numpy(target).unsqueeze(0))
        return {
            "condition": condition_tensor,
            "target": target_tensor,
            "baseline": torch.from_numpy(condition).unsqueeze(0),
            "target_physical": torch.from_numpy(target).unsqueeze(0),
            "coarse_physical": torch.from_numpy(np.asarray(item["low_res"], dtype=np.float32)).unsqueeze(0),
            "high_lats": torch.from_numpy(np.asarray(item["high_lats"], dtype=np.float32)),
            "coarse_lats": torch.from_numpy(np.asarray(item["low_lats"], dtype=np.float32)),
            "event_id": item.get("event_id", f"EV_{idx:04d}"),
            "track_id": item.get("track_id", f"TRK_{idx:04d}"),
            "timestamp": str(item.get("timestamp", "")),
        }


__all__ = ["DiffusionDownscalingDataset", "split_downscaling_samples"]
