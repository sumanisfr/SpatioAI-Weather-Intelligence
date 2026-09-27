"""Event lookup endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import repository
from api.schemas import EventListResponse, EventResponse
from api.services.event_service import EventService

router = APIRouter(prefix="/api/v1/events", tags=["events"])


def service(repo=Depends(repository)):
    return EventService(repo)


@router.get("", response_model=EventListResponse, summary="List tracked events")
def list_events(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, minimum_intensity: Optional[float] = Query(default=None, ge=0), track_id: Optional[str] = None, event_type: Optional[str] = None, events=Depends(service)):
    values = events.list_events(start_time, end_time, minimum_intensity, track_id, event_type)
    return {"events": values, "count": len(values), "source_mode": "synthetic_demo"}


@router.get("/{event_id}", response_model=EventResponse, summary="Get one tracked event")
def get_event(event_id: str, events=Depends(service)):
    value = events.get_event(event_id)
    if value is None:
        raise HTTPException(status_code=404, detail="event not found")
    return value
