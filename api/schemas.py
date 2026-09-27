"""Typed request and response contracts for the SpatioAI API."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventResponse(BaseModel):
    event_id: str
    track_id: Optional[str] = None
    timestamp: datetime
    centroid_lat: float
    centroid_lon: float
    bbox: List[float] = Field(min_length=4, max_length=4)
    area_km2: float = 0.0
    max_intensity: float = 0.0
    mean_intensity: float = 0.0
    model_config = ConfigDict(from_attributes=True)


class EventListResponse(BaseModel):
    events: List[EventResponse]
    count: int
    source_mode: str


class TrackResponse(BaseModel):
    track_id: str
    event_ids: List[str]
    timestamps: List[datetime]
    centroids: List[List[float]]
    bboxes: List[List[float]]
    speed_kmh: Optional[float] = None
    bearing_deg: Optional[float] = None


class TrackListResponse(BaseModel):
    tracks: List[TrackResponse]
    count: int
    source_mode: str


class TrackingPredictRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(min_length=1)
    model: Literal["spatiotemporal_gnn"] = "spatiotemporal_gnn"


class TrackingPredictResponse(BaseModel):
    model: str
    predicted_tracks: List[List[str]]
    edge_predictions: List[Dict[str, Any]]
    source_mode: str


class DownscalingRequest(BaseModel):
    model: Literal["unet", "diffusion", "physics_diffusion"] = "diffusion"
    low_res_field: List[List[float]]
    source_lats: List[float]
    source_lons: List[float]
    target_lats: Optional[List[float]] = None
    target_lons: Optional[List[float]] = None
    event_id: Optional[str] = None
    ensemble_samples: int = Field(default=5, ge=1, le=100)
    seed: int = 42

    @field_validator("low_res_field")
    @classmethod
    def validate_field(cls, value: List[List[float]]) -> List[List[float]]:
        if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
            raise ValueError("low_res_field must be a non-empty rectangular 2D array")
        return value


class DownscalingResponse(BaseModel):
    model: str
    source_mode: str
    target_shape: List[int]
    prediction: Optional[List[List[float]]] = None
    ensemble_mean: Optional[List[List[float]]] = None
    ensemble_std: Optional[List[List[float]]] = None
    quantiles: Dict[str, List[List[float]]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnsembleRequest(BaseModel):
    condition: List[List[float]]
    threshold: Optional[float] = Field(default=None, gt=0)
    num_samples: int = Field(default=5, ge=1, le=100)
    seed: int = 42


class EnsembleResponse(BaseModel):
    model: str
    num_samples: int
    mean: List[List[float]]
    median: List[List[float]]
    std: List[List[float]]
    quantiles: Dict[str, List[List[float]]]
    threshold: Optional[float] = None
    exceedance_probability: Optional[List[List[float]]] = None
    source_mode: str


class RiskRequest(BaseModel):
    event_id: str
    threshold: float = Field(gt=0)
    probability_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ensemble_samples: int = Field(default=5, ge=1, le=100)
    seed: int = 42


class RiskResponse(BaseModel):
    event_id: str
    track_id: Optional[str] = None
    timestamp: datetime
    threshold: float
    max_exceedance_probability: float
    mean_exceedance_probability: float
    affected_area_km2: float
    severity_index: float
    uncertainty: float
    risk_score: float
    geojson: Dict[str, Any]
    source_mode: str


class ModelStatus(BaseModel):
    available: bool
    loaded: bool
    path: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool
    models: Dict[str, ModelStatus]


class VersionResponse(BaseModel):
    service: str
    version: str
    api_version: str


class ErrorResponse(BaseModel):
    detail: str
