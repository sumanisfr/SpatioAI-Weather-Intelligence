"""Tests for Phase 6 conditional diffusion downscaling."""

import numpy as np
import pytest
import torch

from src.downscaling.dataset import split_downscaling_samples
from src.downscaling.diffusion import (
    ConditionalDiffusionUNet,
    DiffusionLoss,
    DiffusionProcess,
    SinusoidalTimeEmbedding,
    inverse_log1p_transform,
    log1p_transform,
)
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.diffusion_sampler import generate, summarize_ensemble
from src.downscaling.synthetic import generate_paired_synthetic_dataset


def test_schedules_and_forward_diffusion_are_stable():
    process = DiffusionProcess(20, "cosine")
    assert torch.all(process.betas > 0)
    assert torch.all(process.betas < 1)
    assert torch.all(process.alpha_cumprod[1:] <= process.alpha_cumprod[:-1])
    x = torch.ones(2, 1, 8, 8)
    noise = torch.full_like(x, 0.25)
    t = torch.tensor([0, 19])
    first = process.q_sample(x, t, noise)
    second = process.q_sample(x, t, noise)
    assert torch.equal(first, second)
    assert first.shape == x.shape and torch.isfinite(first).all()


def test_linear_schedule_and_timestep_embedding():
    process = DiffusionProcess(10, "linear")
    embedding = SinusoidalTimeEmbedding(16)(torch.arange(4))
    assert embedding.shape == (4, 16)
    assert torch.isfinite(embedding).all()
    assert process.beta_schedule == "linear"


def test_transform_round_trip_and_non_negative_inverse():
    precipitation = torch.tensor([0.0, 1.0, 50.0, 100.0])
    restored = inverse_log1p_transform(log1p_transform(precipitation))
    assert torch.allclose(restored, precipitation, atol=1e-5)
    assert torch.all(inverse_log1p_transform(torch.tensor([-100.0, 1000.0])) >= 0)
    assert torch.isfinite(inverse_log1p_transform(torch.tensor([1000.0]))).all()


def test_conditional_model_and_loss_gradients():
    model = ConditionalDiffusionUNet(base_channels=4)
    process = DiffusionProcess(8)
    x = torch.randn(2, 1, 16, 16)
    condition = torch.randn_like(x)
    t = torch.tensor([1, 7])
    output = model(x, t, condition)
    changed = model(x, t, condition + 1.0)
    assert output.shape == x.shape
    assert not torch.allclose(output, changed)
    loss = DiffusionLoss()(output, torch.randn_like(output), x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert process.q_sample(x, t).shape == x.shape


def test_dataset_metadata_and_track_split():
    samples = generate_paired_synthetic_dataset(num_events=4, timesteps_per_event=2, seed=12)
    train, validation, test = split_downscaling_samples(samples, seed=12)
    train_tracks = {item["track_id"] for item in train}
    validation_tracks = {item["track_id"] for item in validation}
    test_tracks = {item["track_id"] for item in test}
    assert not train_tracks & validation_tracks
    assert not train_tracks & test_tracks
    assert not validation_tracks & test_tracks
    item = DiffusionDownscalingDataset(samples)[0]
    assert item["condition"].shape == item["target"].shape
    assert item["event_id"] == samples[0]["event_id"]
    assert item["track_id"] == samples[0]["track_id"]


def test_sampling_reproducibility_and_ensemble_summary():
    process = DiffusionProcess(5, "linear")
    model = ConditionalDiffusionUNet(base_channels=4)
    condition = torch.rand(1, 1, 16, 16)
    first = generate(model, process, condition, num_samples=3, seed=42)
    second = generate(model, process, condition, num_samples=3, seed=42)
    different = generate(model, process, condition, num_samples=3, seed=123)
    assert first.shape == (1, 3, 1, 16, 16)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, different)
    summary = summarize_ensemble(first)
    assert set(summary) == {"mean", "std", "q10", "q50", "q90"}
    assert torch.isfinite(summary["mean"]).all()
