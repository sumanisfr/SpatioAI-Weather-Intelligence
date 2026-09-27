"""Configuration loading for the Phase 9 API."""

from pathlib import Path
from typing import Any, Dict

import yaml


class APISettings:
    def __init__(self, config_path: str = "configs/data.yaml") -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config: Dict[str, Any] = yaml.safe_load(handle) or {}
        api_config = self.config.get("api", {})
        self.host = api_config.get("host", "127.0.0.1")
        self.port = int(api_config.get("port", 8000))
        self.cors_origins = api_config.get("cors_origins", ["http://localhost:5173"])
        self.device = self.config.get("inference", {}).get("device", "auto")
        self.demo_mode = bool(api_config.get("demo_mode", True))
        self.model_paths = {
            "gnn": self.config.get("models", {}).get("gnn_checkpoint", "models/spatiotemporal_gnn_best.pt"),
            "unet": self.config.get("models", {}).get("unet_checkpoint", "models/downscaler_unet_best.pt"),
            "diffusion": self.config.get("models", {}).get("diffusion_checkpoint", "models/diffusion_downscaler_best.pt"),
            "physics_diffusion": self.config.get("models", {}).get("physics_diffusion_checkpoint", "models/diffusion_physics_ablation.pt"),
        }
        self.ensemble_samples = int(self.config.get("ensemble", {}).get("num_samples", 5))
        self.risk_config = self.config.get("risk", {})

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
