"""Risk analysis service over Phase 8 ensemble/risk modules."""

import torch

from src.downscaling.diffusion_sampler import generate
from src.downscaling.diffusion import log1p_transform
from src.ensemble import compute_ensemble_statistics
from src.risk import build_risk_map, event_risk_summary


def _mask_geojson(mask: torch.Tensor, latitudes, longitudes) -> dict:
    features = []
    mask = mask.detach().cpu()
    for row, col in zip(*torch.where(mask)):
        r, c = int(row), int(col)
        lat0 = float(latitudes[r])
        lat1 = float(latitudes[min(r + 1, len(latitudes) - 1)])
        lon0 = float(longitudes[c])
        lon1 = float(longitudes[min(c + 1, len(longitudes) - 1)])
        if lat1 == lat0:
            lat0, lat1 = lat0 - 0.01, lat1 + 0.01
        if lon1 == lon0:
            lon0, lon1 = lon0 - 0.01, lon1 + 0.01
        features.append({"type": "Feature", "properties": {"row": r, "column": c}, "geometry": {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}})
    return {"type": "FeatureCollection", "features": features}


class RiskService:
    def __init__(self, registry, repository) -> None:
        self.registry = registry
        self.repository = repository

    def analyze(self, request):
        sample = self.repository.get_sample(request.event_id)
        event = self.repository.get_event(request.event_id)
        if sample is None or event is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="event not found")
        bundle = self.registry.require("diffusion")
        condition = log1p_transform(torch.from_numpy(sample["low_res"]).float())
        from src.downscaling.interpolation import interpolate_field
        baseline = interpolate_field(sample["low_res"], sample["low_lats"], sample["low_lons"], sample["high_lats"], sample["high_lons"], method="bilinear")
        condition = log1p_transform(torch.from_numpy(baseline).float()).unsqueeze(0).unsqueeze(0).to(self.registry.device)
        samples = generate(bundle["model"], bundle["process"], condition, num_samples=request.ensemble_samples, seed=request.seed)
        risk_map = build_risk_map(samples, request.threshold, sample["high_lats"], sample["high_lons"], request.probability_threshold, metadata=event)
        summary = event_risk_summary(risk_map, request.threshold, metadata=event)
        stats = compute_ensemble_statistics(samples)
        summary.update({
            "max_exceedance_probability": summary.pop("max_probability"),
            "mean_exceedance_probability": summary.pop("mean_probability"),
            "severity_index": float((stats["mean"] / request.threshold).mean()),
            "uncertainty": float(stats["std"].mean()),
            "geojson": _mask_geojson(risk_map["affected_area_mask"][0, 0], sample["high_lats"], sample["high_lons"]),
            "source_mode": "synthetic_demo",
        })
        return summary
