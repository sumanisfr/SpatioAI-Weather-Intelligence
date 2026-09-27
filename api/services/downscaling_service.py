"""Inference services over the existing downscaling modules."""

from typing import Any, Dict

import numpy as np
import torch

from src.downscaling.diffusion import log1p_transform
from src.downscaling.diffusion_sampler import generate
from src.downscaling.grid import create_target_grid
from src.downscaling.interpolation import interpolate_field
from src.ensemble import compute_ensemble_statistics


class DownscalingService:
    def __init__(self, registry, settings) -> None:
        self.registry = registry
        self.settings = settings

    def predict(self, request) -> Dict[str, Any]:
        low_res = np.asarray(request.low_res_field, dtype=np.float32)
        source_lats = np.asarray(request.source_lats, dtype=np.float32)
        source_lons = np.asarray(request.source_lons, dtype=np.float32)
        if low_res.shape != (len(source_lats), len(source_lons)):
            raise ValueError("low_res_field shape must match source_lats and source_lons")
        if request.target_lats is None or request.target_lons is None:
            target_lats, target_lons = create_target_grid((float(source_lats.min()), float(source_lats.max())), (float(source_lons.min()), float(source_lons.max())), target_res_km=5.0, target_shape=(max(16, len(source_lats) * 3), max(16, len(source_lons) * 3)))
        else:
            target_lats = np.asarray(request.target_lats, dtype=np.float32)
            target_lons = np.asarray(request.target_lons, dtype=np.float32)
        baseline = interpolate_field(low_res, source_lats, source_lons, target_lats, target_lons, method="bilinear").astype(np.float32)
        if request.model == "unet":
            predictor = self.registry.require("unet")
            result = predictor.predict_field(low_res, source_lats, source_lons, target_lats, target_lons)
            return {"model": "unet", "source_mode": "synthetic_demo", "target_shape": list(result["prediction"].shape), "prediction": result["prediction"].tolist(), "metadata": {"event_id": request.event_id}}
        model_name = "physics_diffusion" if request.model == "physics_diffusion" else "diffusion"
        bundle = self.registry.require(model_name)
        condition = log1p_transform(torch.from_numpy(baseline).unsqueeze(0).unsqueeze(0)).to(self.registry.device)
        samples = generate(bundle["model"], bundle["process"], condition, num_samples=request.ensemble_samples, seed=request.seed)
        statistics = compute_ensemble_statistics(samples)
        return {
            "model": "physics_diffusion" if request.model == "physics_diffusion" else "conditional_diffusion",
            "source_mode": "synthetic_demo",
            "target_shape": list(baseline.shape),
            "ensemble_mean": statistics["mean"].squeeze().cpu().tolist(),
            "ensemble_std": statistics["std"].squeeze().cpu().tolist(),
            "quantiles": {key: value.squeeze().cpu().tolist() for key, value in statistics.items() if key.startswith("q")},
            "metadata": {"event_id": request.event_id, "num_samples": request.ensemble_samples, "mean_available": True, "uncertainty_available": True},
        }
