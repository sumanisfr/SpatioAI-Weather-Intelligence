"""Track lookup and GNN inference endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import registry, repository
from api.schemas import TrackListResponse, TrackResponse, TrackingPredictRequest, TrackingPredictResponse
from api.services.event_service import EventService
from api.services.tracking_service import TrackingService

router = APIRouter(prefix="/api/v1", tags=["tracking"])


@router.get("/tracks", response_model=TrackListResponse, summary="List tracked trajectories")
def list_tracks(repo=Depends(repository)):
    values = EventService(repo).list_tracks()
    return {"tracks": values, "count": len(values), "source_mode": "synthetic_demo"}


@router.get("/tracks/{track_id}", response_model=TrackResponse, summary="Get one tracked trajectory")
def get_track(track_id: str, repo=Depends(repository)):
    from fastapi import HTTPException
    value = EventService(repo).get_track(track_id)
    if value is None:
        raise HTTPException(status_code=404, detail="track not found")
    return value


@router.post("/tracking/predict", response_model=TrackingPredictResponse, summary="Predict event associations with the GNN")
def predict_tracking(request: TrackingPredictRequest, model_registry=Depends(registry)):
    return TrackingService(model_registry).predict(request)
