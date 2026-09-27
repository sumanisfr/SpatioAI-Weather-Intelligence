"""Lightweight synthetic end-to-end validation for Phase 6."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch

from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionLoss, DiffusionProcess
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
from src.downscaling.diffusion_sampler import generate, summarize_ensemble
from src.downscaling.synthetic import generate_paired_synthetic_dataset


def validate_diffusion_pipeline() -> bool:
    torch.manual_seed(42)
    samples = generate_paired_synthetic_dataset(num_events=2, timesteps_per_event=1, seed=42)
    dataset = DiffusionDownscalingDataset(samples)
    batch = {key: torch.stack([dataset[i][key] for i in range(2)]) for key in ("condition", "target")}
    process = DiffusionProcess(timesteps=8, beta_schedule="cosine")
    model = ConditionalDiffusionUNet(base_channels=4)
    criterion = DiffusionLoss(extreme_enabled=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    losses = []
    for _ in range(4):
        t = torch.randint(0, process.timesteps, (2,))
        noise = torch.randn_like(batch["target"])
        loss = criterion(model(process.q_sample(batch["target"], t, noise), t, batch["condition"]), noise, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"Tiny diffusion loss did not decrease: {losses}"
    generated = generate(model, process, batch["condition"][:1], num_samples=3, seed=42)
    assert generated.shape == (1, 3, 1, batch["target"].shape[-2], batch["target"].shape[-1])
    assert torch.isfinite(generated).all() and torch.all(generated >= 0)
    summary = summarize_ensemble(generated)
    assert summary["mean"].shape == batch["target"][:1].shape
    print("PHASE 6 SYNTHETIC VALIDATION PASSED")
    print(f"Tiny loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Generated samples: {tuple(generated.shape)}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_diffusion_pipeline() else 1)
