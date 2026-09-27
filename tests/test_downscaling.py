"""
Unit and integration tests for SpatioAI Phase 5: 12 km -> ~5 km Extreme-Event Downscaling.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.downscaling.cnn import UNetDownscaler
from src.downscaling.dataset import DownscalingDataset, split_downscaling_samples
from src.downscaling.evaluation import (
    compare_downscaling_methods,
    compute_extreme_downscaling_metrics,
    compute_gradient_error,
    compute_ssim_2d,
)
from src.downscaling.grid import calculate_grid_spacing, create_target_grid, extract_event_crop
from src.downscaling.inference import DownscalingPredictor
from src.downscaling.interpolation import interpolate_field
from src.downscaling.losses import CombinedDownscalingLoss, WeightedExtremeMSELoss, get_downscaling_loss
from src.downscaling.synthetic import (
    downsample_field_to_coarse,
    generate_paired_synthetic_dataset,
    generate_synthetic_storm_field,
)


# --- 1. Grid & Event Crop Tests ---

def test_calculate_grid_spacing():
    lats = np.linspace(10.0, 20.0, 101)  # 0.1 deg ~ 11.1 km
    lons = np.linspace(70.0, 80.0, 101)
    metrics = calculate_grid_spacing(lats, lons)

    assert metrics["dlat_deg"] == pytest.approx(0.1, rel=1e-2)
    assert metrics["dlat_km"] == pytest.approx(11.11, rel=1e-1)
    assert metrics["effective_res_km"] > 0
    assert metrics["dlon_km_mean"] > 0


def test_create_target_grid():
    lat_bounds = (12.0, 16.0)
    lon_bounds = (80.0, 84.0)
    target_lats, target_lons = create_target_grid(lat_bounds, lon_bounds, target_res_km=5.0)

    assert len(target_lats) >= 4
    assert len(target_lons) >= 4
    assert target_lats[0] == pytest.approx(12.0)
    assert target_lats[-1] == pytest.approx(16.0)


def test_extract_event_crop():
    lats = np.linspace(10.0, 30.0, 100)
    lons = np.linspace(65.0, 95.0, 100)
    field = np.zeros((100, 100))
    field[40:60, 40:60] = 50.0  # Synthetic storm

    bbox = (15.0, 20.0, 75.0, 80.0)
    crop = extract_event_crop(field, bbox=bbox, lats=lats, lons=lons, padding_deg=1.0)

    assert crop["crop_data"].ndim == 2
    assert crop["crop_data"].shape[0] > 0
    assert crop["crop_data"].shape[1] > 0
    assert crop["bbox_padded"][0] <= bbox[0]
    assert crop["bbox_padded"][1] >= bbox[1]


# --- 2. Interpolation Baseline Tests ---

def test_interpolation_constant_preservation():
    src_lats = np.linspace(10.0, 20.0, 10)
    src_lons = np.linspace(70.0, 80.0, 10)
    tgt_lats = np.linspace(10.0, 20.0, 25)
    tgt_lons = np.linspace(70.0, 80.0, 25)

    const_val = 37.5
    src_field = np.full((10, 10), const_val)

    nearest = interpolate_field(src_field, src_lats, src_lons, tgt_lats, tgt_lons, method="nearest")
    bilinear = interpolate_field(src_field, src_lats, src_lons, tgt_lats, tgt_lons, method="bilinear")

    assert nearest.shape == (25, 25)
    assert bilinear.shape == (25, 25)
    assert np.allclose(nearest, const_val)
    assert np.allclose(bilinear, const_val)


def test_interpolation_multidim():
    src_lats = np.linspace(10.0, 20.0, 10)
    src_lons = np.linspace(70.0, 80.0, 10)
    tgt_lats = np.linspace(10.0, 20.0, 20)
    tgt_lons = np.linspace(70.0, 80.0, 20)

    field_3d = np.ones((4, 10, 10)) * 12.0
    field_4d = np.ones((2, 3, 10, 10)) * 15.0

    out_3d = interpolate_field(field_3d, src_lats, src_lons, tgt_lats, tgt_lons, method="bilinear")
    out_4d = interpolate_field(field_4d, src_lats, src_lons, tgt_lats, tgt_lons, method="bilinear")

    assert out_3d.shape == (4, 20, 20)
    assert out_4d.shape == (2, 3, 20, 20)


# --- 3. Dataset & Leakage Tests ---

def test_synthetic_data_generation():
    samples = generate_paired_synthetic_dataset(num_events=3, timesteps_per_event=3, seed=42)
    assert len(samples) == 9

    s0 = samples[0]
    assert "low_res" in s0 and "high_res" in s0
    assert s0["low_res"].ndim == 2
    assert s0["high_res"].ndim == 2
    assert np.min(s0["high_res"]) >= 0.0  # Non-negativity


def test_leakage_safe_split():
    samples = generate_paired_synthetic_dataset(num_events=6, timesteps_per_event=4, seed=42)
    train_s, val_s, test_s = split_downscaling_samples(samples, method="track", train_ratio=0.7, val_ratio=0.15, seed=42)

    train_tracks = set(s["track_id"] for s in train_s)
    val_tracks = set(s["track_id"] for s in val_s)
    test_tracks = set(s["track_id"] for s in test_s)

    assert len(train_tracks.intersection(val_tracks)) == 0
    assert len(train_tracks.intersection(test_tracks)) == 0
    assert len(val_tracks.intersection(test_tracks)) == 0


def test_downscaling_dataset_item():
    samples = generate_paired_synthetic_dataset(num_events=2, timesteps_per_event=2, seed=42)
    ds = DownscalingDataset(samples, interp_method="bilinear")

    assert len(ds) == 4
    item = ds[0]

    assert isinstance(item["x"], torch.Tensor)
    assert isinstance(item["y"], torch.Tensor)
    assert item["x"].shape == item["y"].shape
    assert item["x"].ndim == 3  # [1, H, W]


# --- 4. U-Net Model & Loss Tests ---

def test_unet_forward_and_shapes():
    model = UNetDownscaler(in_channels=1, out_channels=1, base_channels=16, residual_learning=True, output_activation="softplus")
    x = torch.randn(2, 1, 32, 32).abs()
    out = model(x)

    assert out.shape == (2, 1, 32, 32)
    assert torch.all(out >= 0.0)  # Softplus non-negative constraint


def test_unet_gradients():
    model = UNetDownscaler(in_channels=1, out_channels=1, base_channels=16, residual_learning=True)
    x = torch.randn(2, 1, 24, 24)
    target = torch.randn(2, 1, 24, 24).abs()

    loss_fn = get_downscaling_loss(loss_name="weighted_mse", extreme_percentile=90.0, extreme_weight=2.0)
    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_extreme_weighted_loss():
    loss_fn = WeightedExtremeMSELoss(extreme_percentile=90.0, extreme_weight=5.0)
    pred = torch.tensor([[[[10.0, 50.0]]]])
    target = torch.tensor([[[[10.0, 100.0]]]])

    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


# --- 5. Evaluation Metrics Tests ---

def test_evaluation_metrics():
    target = np.array([[10.0, 20.0], [50.0, 100.0]])
    pred = np.array([[11.0, 22.0], [48.0, 95.0]])

    metrics = compute_extreme_downscaling_metrics(pred, target, percentiles=(90.0, 95.0))

    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0
    assert metrics["correlation"] > 0.90
    assert "bias_p95" in metrics
    assert "peak_error" in metrics
    assert "rel_peak_error" in metrics


def test_compare_downscaling_methods():
    target = np.ones((20, 20)) * 50.0
    near = np.ones((20, 20)) * 48.0
    bili = np.ones((20, 20)) * 49.5
    unet = np.ones((20, 20)) * 50.0

    preds = {"Nearest": near, "Bilinear": bili, "UNet": unet}
    comp = compare_downscaling_methods(preds, target)

    assert comp["UNet"]["mae"] == pytest.approx(0.0)
    assert comp["Bilinear"]["mae"] < comp["Nearest"]["mae"]


# --- 6. End-to-End Pipeline & Sanity Overfit Test ---

def test_tiny_overfit_sanity():
    samples = generate_paired_synthetic_dataset(num_events=1, timesteps_per_event=2, seed=77)
    ds = DownscalingDataset(samples)
    x = torch.stack([ds[i]["x"] for i in range(len(ds))])
    y = torch.stack([ds[i]["y"] for i in range(len(ds))])

    model = UNetDownscaler(in_channels=1, out_channels=1, base_channels=16, residual_learning=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = get_downscaling_loss(loss_name="weighted_mse", extreme_weight=1.0)

    initial_loss = crit(model(x), y).item()
    for _ in range(80):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        optimizer.step()

    final_loss = crit(model(x), y).item()
    assert final_loss < initial_loss * 0.40, f"Overfit sanity failed: {initial_loss} -> {final_loss}"


def test_downscaling_predictor_inference():
    predictor = DownscalingPredictor(device="cpu")
    lats = np.linspace(15.0, 20.0, 20)
    lons = np.linspace(75.0, 80.0, 20)
    full_field = np.random.uniform(0, 50, (20, 20)).astype(np.float32)

    res = predictor.predict_event_crop(
        full_field=full_field,
        full_lats=lats,
        full_lons=lons,
        event_bbox=(16.0, 18.0, 76.0, 78.0),
        padding_deg=0.5,
        target_res_km=5.0,
    )

    assert "prediction" in res
    assert "baseline_bilinear" in res
    assert "baseline_nearest" in res
    assert res["prediction"].shape == res["baseline_bilinear"].shape
    assert np.all(res["prediction"] >= 0.0)
