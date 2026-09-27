"""
src/downscaling module initialization.
Exposes key classes and functions for Phase 5 Downscaling.
"""

from src.downscaling.cnn import DoubleConv, DownBlock, UNetDownscaler, UpBlock
from src.downscaling.dataset import DownscalingDataset, split_downscaling_samples
from src.downscaling.diffusion import (
    ConditionalDiffusionUNet,
    DiffusionLoss,
    DiffusionProcess,
    SinusoidalTimeEmbedding,
    inverse_log1p_transform,
    log1p_transform,
)
from src.downscaling.diffusion_dataset import DiffusionDownscalingDataset
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

__all__ = [
    "calculate_grid_spacing",
    "create_target_grid",
    "extract_event_crop",
    "interpolate_field",
    "generate_synthetic_storm_field",
    "downsample_field_to_coarse",
    "generate_paired_synthetic_dataset",
    "DownscalingDataset",
    "split_downscaling_samples",
    "DiffusionDownscalingDataset",
    "DiffusionProcess",
    "ConditionalDiffusionUNet",
    "SinusoidalTimeEmbedding",
    "DiffusionLoss",
    "log1p_transform",
    "inverse_log1p_transform",
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "UNetDownscaler",
    "WeightedExtremeMSELoss",
    "CombinedDownscalingLoss",
    "get_downscaling_loss",
    "compute_ssim_2d",
    "compute_gradient_error",
    "compute_extreme_downscaling_metrics",
    "compare_downscaling_methods",
    "DownscalingPredictor",
]
