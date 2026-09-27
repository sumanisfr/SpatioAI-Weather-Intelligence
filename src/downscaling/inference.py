"""
Inference pipeline for extreme-event downscaling in SpatioAI Phase 5.
Supports:
  Event Track / BBox -> Dynamic Crop -> Baseline Interpolation -> U-Net Prediction -> Output Array & Metadata.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from src.downscaling.cnn import UNetDownscaler
from src.downscaling.grid import calculate_grid_spacing, create_target_grid, extract_event_crop
from src.downscaling.interpolation import interpolate_field
from src.utils.logger import get_logger

logger = get_logger("DownscalingInference")


class DownscalingPredictor:
    """
    Inference manager for U-Net downscaling model.
    Loads checkpoint, performs input preparation and executes fine-scale spatial prediction.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[UNetDownscaler] = None,
        device: Optional[Union[str, torch.device]] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        residual_learning: bool = True,
        output_activation: str = "softplus",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(
                Path(checkpoint_path),
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                residual_learning=residual_learning,
                output_activation=output_activation,
            )
        else:
            self.model = UNetDownscaler(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                residual_learning=residual_learning,
                output_activation=output_activation,
            ).to(self.device)

        self.model.eval()

    def _load_model_from_checkpoint(
        self,
        ckpt_path: Path,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        residual_learning: bool = True,
        output_activation: str = "softplus",
    ) -> UNetDownscaler:
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint {ckpt_path} not found; initializing fresh model.")
            return UNetDownscaler(
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                residual_learning=residual_learning,
                output_activation=output_activation,
            ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_config = checkpoint.get("config", {}).get("model", {})

        model = UNetDownscaler(
            in_channels=model_config.get("in_channels", in_channels),
            out_channels=model_config.get("out_channels", out_channels),
            base_channels=model_config.get("base_channels", base_channels),
            residual_learning=model_config.get("residual_learning", residual_learning),
            output_activation=model_config.get("output_activation", output_activation),
        ).to(self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded downscaler weights from {ckpt_path}")
        return model

    def predict_field(
        self,
        low_res_field: np.ndarray,
        source_lats: np.ndarray,
        source_lons: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Downscale a single 2D coarse field onto specified target grid.

        Returns:
            Dict containing 'prediction', 'baseline_bilinear', 'baseline_nearest', 'target_lats', 'target_lons'.
        """
        # 1. Classical Baselines
        baseline_bilinear = interpolate_field(
            low_res_field, source_lats, source_lons, target_lats, target_lons, method="bilinear"
        ).astype(np.float32)

        baseline_nearest = interpolate_field(
            low_res_field, source_lats, source_lons, target_lats, target_lons, method="nearest"
        ).astype(np.float32)

        # 2. U-Net Forward Pass
        x_tensor = torch.from_numpy(baseline_bilinear).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(x_tensor)
            pred_2d = pred_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        return {
            "prediction": pred_2d,
            "baseline_bilinear": baseline_bilinear,
            "baseline_nearest": baseline_nearest,
            "target_lats": target_lats,
            "target_lons": target_lons,
            "grid_metrics": calculate_grid_spacing(target_lats, target_lons),
        }

    def predict_event_crop(
        self,
        full_field: np.ndarray,
        full_lats: np.ndarray,
        full_lons: np.ndarray,
        event_bbox: Tuple[float, float, float, float],
        padding_deg: float = 1.0,
        target_res_km: float = 5.0,
    ) -> Dict[str, Any]:
        """
        End-to-end inference from full coarse weather field and event bounding box:
        Crop -> High-Res Target Grid -> Bilinear -> U-Net.
        """
        # Crop
        crop_res = extract_event_crop(
            field_or_ds=full_field,
            bbox=event_bbox,
            lats=full_lats,
            lons=full_lons,
            padding_deg=padding_deg,
        )
        crop_data = crop_res["crop_data"]
        crop_lats = crop_res["lats"]
        crop_lons = crop_res["lons"]

        # Create target grid (~5 km)
        lat_bounds = (float(np.min(crop_lats)), float(np.max(crop_lats)))
        lon_bounds = (float(np.min(crop_lons)), float(np.max(crop_lons)))
        target_lats, target_lons = create_target_grid(lat_bounds, lon_bounds, target_res_km=target_res_km)

        # Predict
        res = self.predict_field(crop_data, crop_lats, crop_lons, target_lats, target_lons)
        res["crop_metadata"] = crop_res["metadata"]
        res["low_res_crop"] = crop_data
        res["low_lats"] = crop_lats
        res["low_lons"] = crop_lons
        return res
