"""Synthetic end-to-end validation for Phase 8 uncertainty and risk."""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.diffusion_sampler import generate
from src.downscaling.synthetic import generate_paired_synthetic_dataset
from src.ensemble import compute_ensemble_statistics, uncertainty_map
from src.risk import build_risk_map, event_risk_summary, temporal_risk_summary


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def validate_risk_pipeline(output_dir: str = "data/outputs/risk", seed: int = 42) -> bool:
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    maps_path = output_path / "risk_maps"
    stats_path = output_path / "ensemble_statistics"
    probability_path = output_path / "exceedance_probability"
    for path in (figures_path, maps_path, stats_path, probability_path):
        path.mkdir(parents=True, exist_ok=True)

    sample = generate_paired_synthetic_dataset(num_events=1, timesteps_per_event=1, seed=seed)[0]
    item = DiffusionDownscalingDataset([sample])[0]
    model = ConditionalDiffusionUNet(base_channels=4)
    process = DiffusionProcess(timesteps=8, beta_schedule="linear")
    ensemble = generate(model, process, item["condition"].unsqueeze(0), num_samples=5, seed=seed)
    threshold = float(torch.quantile(item["target_physical"], 0.95))
    metadata = {
        "event_id": sample["event_id"],
        "track_id": sample["track_id"],
        "timestamp": sample["timestamp"],
        "centroid_lat": sample["centroid_lat"],
        "centroid_lon": sample["centroid_lon"],
        "bbox": [float(sample["high_lats"].min()), float(sample["high_lats"].max()), float(sample["high_lons"].min()), float(sample["high_lons"].max())],
    }
    risk_map = build_risk_map(ensemble, threshold, sample["high_lats"], sample["high_lons"], metadata=metadata)
    summary = event_risk_summary(risk_map, threshold, metadata)
    assert torch.isfinite(ensemble).all()
    assert torch.all((risk_map["exceedance_probability"] >= 0) & (risk_map["exceedance_probability"] <= 1))
    assert torch.all(risk_map["uncertainty"] >= 0)
    assert summary["affected_area_km2"] >= 0
    assert summary["event_id"] == sample["event_id"] and summary["track_id"] == sample["track_id"]

    stats = compute_ensemble_statistics(ensemble)
    torch.save({key: value.cpu() for key, value in stats.items()}, stats_path / f"{sample['event_id']}_statistics.pt")
    torch.save(risk_map["exceedance_probability"].cpu(), probability_path / f"{sample['event_id']}_probability.pt")
    torch.save({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in risk_map.items() if isinstance(value, torch.Tensor)}, maps_path / f"{sample['event_id']}_risk_map.pt")
    with open(output_path / "event_risk_summary.json", "w") as handle:
        json.dump(_json_ready(summary), handle, indent=2)
    pd.DataFrame([summary]).to_csv(output_path / "event_risk_summary.csv", index=False)

    mean_field = stats["mean"][0, 0]
    std_field = stats["std"][0, 0]
    probability_field = risk_map["exceedance_probability"][0, 0]
    mask = risk_map["affected_area_mask"][0, 0]
    fields = [("Ensemble mean", mean_field), ("Ensemble std", std_field), ("Q95", stats["q95"][0, 0]), ("Exceedance probability", probability_field), ("Affected-area mask", mask.float()), ("Risk score", risk_map["severity"][0, 0])]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, (title, field) in zip(axes.flat, fields):
        image = axis.imshow(field.detach().cpu(), origin="lower", cmap="Blues", extent=[sample["high_lons"].min(), sample["high_lons"].max(), sample["high_lats"].min(), sample["high_lats"].max()])
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(figures_path / "risk_maps.png", dpi=150)
    plt.close(figure)

    trajectory = pd.DataFrame([{"timestamp": sample["timestamp"], "centroid_lat": sample["centroid_lat"], "centroid_lon": sample["centroid_lon"]}])
    trajectory_figure, trajectory_axis = plt.subplots(figsize=(6, 5))
    trajectory_axis.plot(trajectory["centroid_lon"], trajectory["centroid_lat"], marker="o", label="tracked centroid")
    trajectory_axis.contourf(sample["high_lons"], sample["high_lats"], probability_field.detach().cpu(), levels=5, alpha=0.5, cmap="Reds")
    trajectory_axis.set_title("Tracked event centroid and empirical risk footprint")
    trajectory_axis.set_xlabel("longitude")
    trajectory_axis.set_ylabel("latitude")
    trajectory_axis.legend()
    trajectory_figure.tight_layout()
    trajectory_figure.savefig(figures_path / "trajectory_risk_footprint.png", dpi=150)
    plt.close(trajectory_figure)

    assert len(temporal_risk_summary([summary])) == 1
    print("PHASE 8 SYNTHETIC RISK VALIDATION PASSED")
    print(f"Ensemble shape: {tuple(ensemble.shape)}")
    print(f"Threshold: {threshold:.4f}; affected area: {summary['affected_area_km2']:.2f} km2")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_risk_pipeline() else 1)
