"""Startup model registry with explicit availability states."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from api.config import APISettings
from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess
from src.downscaling.inference import DownscalingPredictor


class ModelRegistry:
    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.resolved_device)
        self.models: Dict[str, Any] = {}
        self.statuses: Dict[str, Dict[str, Any]] = {}

    def _record(self, name: str, path: str, loaded: bool, error: Optional[str] = None) -> None:
        self.statuses[name] = {"available": Path(path).exists(), "loaded": loaded, "path": path, "error": error}

    def load_all(self) -> None:
        self._load_unet()
        self._load_diffusion("diffusion", self.settings.model_paths["diffusion"])
        self._load_diffusion("physics_diffusion", self.settings.model_paths["physics_diffusion"])
        self._load_gnn()

    def _load_unet(self) -> None:
        path = self.settings.model_paths["unet"]
        try:
            if not Path(path).exists():
                raise FileNotFoundError("checkpoint unavailable")
            self.models["unet"] = DownscalingPredictor(checkpoint_path=path, device=self.device)
            self._record("unet", path, True)
        except Exception as exc:
            self._record("unet", path, False, str(exc))

    def _load_diffusion(self, name: str, path: str) -> None:
        try:
            if not Path(path).exists():
                raise FileNotFoundError("checkpoint unavailable")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            model_config = config.get("model", {})
            model = ConditionalDiffusionUNet(
                base_channels=int(model_config.get("base_channels", 32)),
                condition_channels=int(model_config.get("condition_channels", 1)),
                target_channels=int(model_config.get("target_channels", 1)),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            process = DiffusionProcess(int(config.get("timesteps", 100)), config.get("beta_schedule", "cosine")).to(self.device)
            self.models[name] = {"model": model, "process": process}
            self._record(name, path, True)
        except Exception as exc:
            self._record(name, path, False, str(exc))

    def _load_gnn(self) -> None:
        path = self.settings.model_paths["gnn"]
        try:
            from src.gnn.inference import GNNPredictor
            self.models["gnn"] = GNNPredictor(path, device=str(self.device))
            self._record("gnn", path, True)
        except Exception as exc:
            self._record("gnn", path, False, str(exc))

    def require(self, name: str) -> Any:
        status = self.statuses.get(name, {"available": False, "loaded": False})
        if not status.get("loaded"):
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=f"Model '{name}' is unavailable")
        return self.models[name]

    @property
    def all_loaded(self) -> bool:
        return bool(self.statuses) and all(status["loaded"] for status in self.statuses.values() if status["available"])
