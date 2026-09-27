"""Event risk analysis endpoint."""

from fastapi import APIRouter, Depends

from api.dependencies import registry, repository
from api.schemas import RiskRequest, RiskResponse
from api.services.risk_service import RiskService

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.post("/analyze", response_model=RiskResponse, summary="Analyze empirical threshold exceedance risk")
def analyze_risk(request: RiskRequest, model_registry=Depends(registry), repo=Depends(repository)):
    return RiskService(model_registry, repo).analyze(request)
