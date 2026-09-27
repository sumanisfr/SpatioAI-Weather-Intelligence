"""Conditional diffusion utilities for precipitation downscaling (Phase 6).

The implementation operates on transformed precipitation fields and conditions
every denoising step on the Phase 5 bilinear high-resolution field.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def log1p_transform(precipitation: torch.Tensor) -> torch.Tensor:
    """Map non-negative precipitation to a less skewed training space."""
    return torch.log1p(torch.clamp(precipitation, min=0.0))


def inverse_log1p_transform(transformed: torch.Tensor) -> torch.Tensor:
    """Map transformed precipitation back to non-negative physical units."""
    # Keep an untrained/unstable sampler from overflowing expm1 during validation.
    return torch.expm1(torch.clamp(transformed, max=20.0)).clamp_min(0.0)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return values.gather(0, timesteps.long()).reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Return the cosine DDPM beta schedule from Nichol and Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Return a linearly increasing beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DiffusionProcess(nn.Module):
    """Forward and reverse utilities for a fixed DDPM schedule."""

    def __init__(self, timesteps: int = 100, beta_schedule: str = "cosine") -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        schedule = beta_schedule.lower()
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError("beta_schedule must be 'linear' or 'cosine'")

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.beta_schedule = schedule
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("posterior_variance", betas * (1.0 - torch.cat([torch.ones(1), alpha_cumprod[:-1]])) / (1.0 - alpha_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t from x_0 using the closed-form forward process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("noise and x_start must have the same shape")
        t = t.to(device=x_start.device, dtype=torch.long)
        if torch.any((t < 0) | (t >= self.timesteps)):
            raise ValueError("t contains an invalid diffusion timestep")
        return _extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start + _extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        t = t.to(device=x_t.device, dtype=torch.long)
        return (_extract(self.sqrt_alpha_cumprod, t, x_t.shape).reciprocal() * x_t - _extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape) * _extract(self.sqrt_alpha_cumprod, t, x_t.shape).reciprocal() * predicted_noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Take one reverse DDPM step, conditioning the model at every step."""
        predicted_noise = model(x_t, t, condition)
        beta_t = _extract(self.betas, t, x_t.shape)
        alpha_t = _extract(self.alphas, t, x_t.shape)
        alpha_bar_t = _extract(self.alpha_cumprod, t, x_t.shape)
        mean = (x_t - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
        if torch.all(t == 0):
            return mean
        variance = _extract(self.posterior_variance, t, x_t.shape).clamp_min(1e-20)
        return mean + torch.sqrt(variance) * torch.randn_like(x_t)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, embedding_dim: int, output_dim: Optional[int] = None) -> None:
        super().__init__()
        output_dim = output_dim or embedding_dim
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1]))
        return self.mlp(embedding)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time = nn.Linear(time_dim, out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        x = x + self.time(time_embedding)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(x)))


class ConditionalDiffusionUNet(nn.Module):
    """Compact U-Net predicting noise from noisy target and interpolated condition."""

    def __init__(self, base_channels: int = 32, condition_channels: int = 1, target_channels: int = 1, time_dim: Optional[int] = None) -> None:
        super().__init__()
        time_dim = time_dim or base_channels * 4
        self.condition_channels = condition_channels
        self.target_channels = target_channels
        self.time_embedding = SinusoidalTimeEmbedding(time_dim, time_dim)
        self.enc1 = _TimeBlock(target_channels + condition_channels, base_channels, time_dim)
        self.enc2 = _TimeBlock(base_channels, base_channels * 2, time_dim)
        self.mid = _TimeBlock(base_channels * 2, base_channels * 4, time_dim)
        self.dec2 = _TimeBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.dec1 = _TimeBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, target_channels, 1)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x_t.shape != condition.shape[:1] + (self.condition_channels,) + x_t.shape[2:] and condition.shape[1] != self.condition_channels:
            raise ValueError("condition has an unexpected channel count")
        if x_t.shape[0] != condition.shape[0] or x_t.shape[2:] != condition.shape[2:]:
            raise ValueError("x_t and condition must share batch and spatial dimensions")
        time_embedding = self.time_embedding(timesteps)
        e1 = self.enc1(torch.cat([x_t, condition], dim=1), time_embedding)
        e2 = self.enc2(F.avg_pool2d(e1, 2), time_embedding)
        mid = self.mid(F.avg_pool2d(e2, 2), time_embedding)
        d2 = F.interpolate(mid, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), time_embedding)
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), time_embedding)
        return self.out(d1)


class DiffusionLoss(nn.Module):
    """Noise-prediction MSE with optional target-extreme weighting."""

    def __init__(self, extreme_enabled: bool = True, extreme_percentile: float = 95.0, extreme_weight: float = 2.0) -> None:
        super().__init__()
        self.extreme_enabled = extreme_enabled
        self.extreme_percentile = extreme_percentile
        self.extreme_weight = extreme_weight

    def forward(self, predicted_noise: torch.Tensor, noise: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        squared_error = (predicted_noise - noise).square()
        if self.extreme_enabled and target is not None:
            threshold = torch.quantile(target.detach().flatten(1), self.extreme_percentile / 100.0, dim=1, keepdim=True)
            weights = torch.ones_like(target)
            weights = torch.where(target >= threshold.view(-1, 1, 1, 1), self.extreme_weight, weights)
            return (squared_error * weights).mean()
        return squared_error.mean()


@torch.no_grad()
def sample_diffusion(model: nn.Module, process: DiffusionProcess, condition: torch.Tensor, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
    """Generate transformed-space conditional samples with deterministic seeding."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=condition.device).manual_seed(seed)
    samples = []
    for _ in range(num_samples):
        x = torch.randn(condition.shape[0], process_model_channels(model), *condition.shape[2:], device=condition.device, generator=generator)
        for step in range(process.timesteps - 1, -1, -1):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            predicted_noise = model(x, t, condition)
            beta = _extract(process.betas, t, x.shape)
            alpha = _extract(process.alphas, t, x.shape)
            alpha_bar = _extract(process.alpha_cumprod, t, x.shape)
            mean = (x - beta * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if step > 0:
                variance = _extract(process.posterior_variance, t, x.shape).clamp_min(1e-20)
                noise = torch.randn(x.shape, device=x.device, generator=generator)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        samples.append(x)
    return torch.stack(samples, dim=1)


def process_model_channels(model: nn.Module) -> int:
    return int(getattr(model, "target_channels", 1))
