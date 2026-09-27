"""Health and version endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import registry
from api.schemas import HealthResponse, ModelStatus, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Check API and model readiness")
def health(model_registry=Depends(registry)):
    return {"status": "ok", "service": "SpatioAI", "version": "0.9.0", "models_loaded": model_registry.all_loaded, "models": model_registry.statuses}


@router.get("/version", response_model=VersionResponse, summary="Get API version")
def version():
    return {"service": "SpatioAI", "version": "0.9.0", "api_version": "v1"}
