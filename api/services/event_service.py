"""Event and track lookup services."""

from datetime import datetime
from typing import Optional

from api.services.repository import DemoRepository


class EventService:
    def __init__(self, repository: DemoRepository) -> None:
        self.repository = repository

    def list_events(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, minimum_intensity: Optional[float] = None, track_id: Optional[str] = None, event_type: Optional[str] = None):
        events = self.repository.list_events()
        result = []
        for event in events:
            timestamp = datetime.fromisoformat(event["timestamp"])
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            if minimum_intensity is not None and event["max_intensity"] < minimum_intensity:
                continue
            if track_id and event["track_id"] != track_id:
                continue
            if event_type and event.get("event_type") != event_type:
                continue
            result.append(event)
        return result

    def get_event(self, event_id: str):
        return self.repository.get_event(event_id)

    def list_tracks(self):
        return self.repository.list_tracks()

    def get_track(self, track_id: str):
        return self.repository.get_track(track_id)
