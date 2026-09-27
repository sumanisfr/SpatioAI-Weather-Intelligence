"""Small deterministic demo repository for API development and tests."""

from collections import defaultdict
from typing import Any, Dict, List

from src.downscaling.synthetic import generate_paired_synthetic_dataset


class DemoRepository:
    def __init__(self, seed: int = 42) -> None:
        self.samples = generate_paired_synthetic_dataset(num_events=4, timesteps_per_event=2, seed=seed)
        self.events: List[Dict[str, Any]] = []
        for sample in self.samples:
            self.events.append({
                "event_id": sample["event_id"],
                "track_id": sample["track_id"],
                "timestamp": sample["timestamp"],
                "centroid_lat": sample["centroid_lat"],
                "centroid_lon": sample["centroid_lon"],
                "bbox": [float(sample["high_lats"].min()), float(sample["high_lats"].max()), float(sample["high_lons"].min()), float(sample["high_lons"].max())],
                "area_km2": float(sample["high_lats"].size * sample["high_lons"].size),
                "max_intensity": sample["max_intensity_target"],
                "mean_intensity": float(sample["high_res"].mean()),
            })
        self.by_event = {event["event_id"]: event for event in self.events}
        self.sample_by_event = {sample["event_id"]: sample for sample in self.samples}

    def list_events(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def get_event(self, event_id: str) -> Dict[str, Any] | None:
        return self.by_event.get(event_id)

    def get_sample(self, event_id: str) -> Dict[str, Any] | None:
        return self.sample_by_event.get(event_id)

    def list_tracks(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            grouped[event["track_id"]].append(event)
        tracks = []
        for track_id, events in grouped.items():
            events = sorted(events, key=lambda item: item["timestamp"])
            tracks.append({
                "track_id": track_id,
                "event_ids": [item["event_id"] for item in events],
                "timestamps": [item["timestamp"] for item in events],
                "centroids": [[item["centroid_lat"], item["centroid_lon"]] for item in events],
                "bboxes": [item["bbox"] for item in events],
                "speed_kmh": None,
                "bearing_deg": None,
            })
        return tracks

    def get_track(self, track_id: str) -> Dict[str, Any] | None:
        return next((track for track in self.list_tracks() if track["track_id"] == track_id), None)
