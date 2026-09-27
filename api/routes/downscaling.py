"""Downscaling inference endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import registry
from api.schemas import DownscalingRequest, DownscalingResponse
from api.services.downscaling_service import DownscalingService

router = APIRouter(prefix="/api/v1/downscaling", tags=["downscaling"])


@router.post("/predict", response_model=DownscalingResponse, summary="Downscale a coarse precipitation field")
def predict_downscaling(request: DownscalingRequest, model_registry=Depends(registry)):
    try:
        return DownscalingService(model_registry, None).predict(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
