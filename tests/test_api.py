"""Integration tests for the Phase 9 FastAPI backend."""

import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app
from src.downscaling.cnn import UNetDownscaler
from src.downscaling.diffusion import ConditionalDiffusionUNet, DiffusionProcess
from src.downscaling.inference import DownscalingPredictor
from src.gnn.dataset import generate_synthetic_event_sequence


class DummyGNN:
    def predict_events(self, events):
        ids = [e.get("event_id", f"ev_{i}") for i, e in enumerate(events)]
        return {"predicted_tracks": [ids] if ids else [], "edge_predictions": []}


@pytest.fixture(scope="module")
def client():
    test_client = TestClient(app)
    test_client.__enter__()
    registry = app.state.registry

    # 1. Diffusion mock
    registry.models["diffusion"] = {
        "model": ConditionalDiffusionUNet(base_channels=4),
        "process": DiffusionProcess(2, "linear"),
    }
    registry.models["diffusion"]["model"].eval()
    registry.statuses["diffusion"] = {"available": True, "loaded": True, "path": "mock"}

    # 2. U-Net mock
    registry.models["unet"] = DownscalingPredictor(model=UNetDownscaler(base_channels=4))
    registry.statuses["unet"] = {"available": True, "loaded": True, "path": "mock"}

    # 3. GNN mock
    registry.models["gnn"] = DummyGNN()
    registry.statuses["gnn"] = {"available": True, "loaded": True, "path": "mock"}

    yield test_client
    test_client.__exit__(None, None, None)


def test_health_root_version_and_docs(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers.get("X-Request-ID")
    assert health.json()["status"] == "ok"
    assert "unet" in health.json()["models"]
    assert client.get("/").status_code == 200
    assert client.get("/version").json()["api_version"] == "v1"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_event_and_track_endpoints(client):
    events = client.get("/api/v1/events")
    assert events.status_code == 200
    event = events.json()["events"][0]
    assert client.get(f"/api/v1/events/{event['event_id']}").status_code == 200
    assert client.get("/api/v1/events/unknown-event").status_code == 404
    tracks = client.get("/api/v1/tracks")
    assert tracks.status_code == 200
    track_id = tracks.json()["tracks"][0]["track_id"]
    assert client.get(f"/api/v1/tracks/{track_id}").status_code == 200
    assert client.get("/api/v1/tracks/unknown-track").status_code == 404


def test_downscaling_and_ensemble_endpoints(client):
    field = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]]
    payload = {"model": "unet", "low_res_field": field, "source_lats": [0, 1, 2, 3], "source_lons": [70, 71, 72, 73]}
    downscaled = client.post("/api/v1/downscaling/predict", json=payload)
    assert downscaled.status_code == 200
    assert downscaled.json()["target_shape"]
    ensemble = client.post("/api/v1/ensemble/predict", json={"condition": field, "num_samples": 2, "threshold": 5})
    assert ensemble.status_code == 200
    assert ensemble.json()["num_samples"] == 2
    assert ensemble.json()["exceedance_probability"] is not None


def test_risk_endpoint_and_validation(client):
    event = client.get("/api/v1/events").json()["events"][0]
    response = client.post("/api/v1/risk/analyze", json={"event_id": event["event_id"], "threshold": 50, "ensemble_samples": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == event["event_id"]
    assert 0 <= body["max_exceedance_probability"] <= 1
    assert body["geojson"]["type"] == "FeatureCollection"
    assert client.post("/api/v1/risk/analyze", json={"event_id": event["event_id"], "threshold": -1}).status_code == 422
    assert client.post("/api/v1/risk/analyze", json={"event_id": event["event_id"], "threshold": 10, "probability_threshold": 2}).status_code == 422
    assert client.post("/api/v1/risk/analyze", json={"event_id": "missing", "threshold": 10}).status_code == 404


def test_unavailable_model_is_explicit(client):
    response = client.post("/api/v1/downscaling/predict", json={"model": "physics_diffusion", "low_res_field": [[1, 2], [3, 4]], "source_lats": [0, 1], "source_lons": [70, 71]})
    assert response.status_code == 503


def test_gnn_tracking_endpoint(client):
    events, _ = generate_synthetic_event_sequence(num_storms=1, timesteps_per_storm=3, random_drop_rate=0, seed=4)
    response = client.post("/api/v1/tracking/predict", json={"events": events})
    assert response.status_code == 200
    assert response.json()["model"] == "spatiotemporal_gnn"
    assert response.json()["predicted_tracks"]
