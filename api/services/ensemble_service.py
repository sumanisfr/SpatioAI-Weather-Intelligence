"""Conditional ensemble inference service."""

import numpy as np
import torch

from src.downscaling.diffusion import log1p_transform
from src.downscaling.diffusion_sampler import generate
from src.downscaling.interpolation import interpolate_field
from src.ensemble import compute_ensemble_statistics, threshold_exceedance_probability


class EnsembleService:
    def __init__(self, registry) -> None:
        self.registry = registry

    def predict(self, request):
        field = np.asarray(request.condition, dtype=np.float32)
        if field.ndim != 2 or not field.size:
            raise ValueError("condition must be a non-empty 2D field")
        bundle = self.registry.require("diffusion")
        condition = log1p_transform(torch.from_numpy(field).unsqueeze(0).unsqueeze(0)).to(self.registry.device)
        samples = generate(bundle["model"], bundle["process"], condition, num_samples=request.num_samples, seed=request.seed)
        statistics = compute_ensemble_statistics(samples)
        result = {
            "model": "conditional_diffusion",
            "num_samples": request.num_samples,
            "mean": statistics["mean"].squeeze().cpu().tolist(),
            "median": statistics["median"].squeeze().cpu().tolist(),
            "std": statistics["std"].squeeze().cpu().tolist(),
            "quantiles": {key: value.squeeze().cpu().tolist() for key, value in statistics.items() if key.startswith("q")},
            "source_mode": "synthetic_demo",
        }
        if request.threshold is not None:
            result["threshold"] = request.threshold
            result["exceedance_probability"] = threshold_exceedance_probability(samples, request.threshold).squeeze().cpu().tolist()
        return result
