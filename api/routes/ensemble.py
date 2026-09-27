"""Conditional ensemble endpoint."""

from fastapi import APIRouter, Depends

from api.dependencies import registry
from api.schemas import EnsembleRequest, EnsembleResponse
from api.services.ensemble_service import EnsembleService

router = APIRouter(prefix="/api/v1/ensemble", tags=["ensemble"])


@router.post("/predict", response_model=EnsembleResponse, summary="Generate conditional ensemble statistics")
def predict_ensemble(request: EnsembleRequest, model_registry=Depends(registry)):
    return EnsembleService(model_registry).predict(request)
