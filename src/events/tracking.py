"""
Temporal Extreme Weather Event Tracking and Lifecycle Management for SpatioAI Phase 3.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.events.association import EventAssociationConfig, EventCandidateMatcher
from src.utils.geodesics import (
    calculate_bearing_deg,
    calculate_speed_kmh,
    haversine_distance_km,
)
from src.utils.logger import get_logger

logger = get_logger("EventTracking")


class KalmanMotionModel:
    """
    Optional 2D constant-velocity kinematic Kalman Filter for geographic centroid tracking.
    State vector: [lat, lon, v_lat, v_lon]^T in deg and deg/hour.
    """

    def __init__(self, init_lat: float, init_lon: float, process_noise: float = 1e-3, meas_noise: float = 1e-2):
        self.state = np.array([init_lat, init_lon, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 0.1
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * meas_noise

    def predict(self, dt_hours: float = 6.0) -> Tuple[float, float]:
        """Predict next position given time step."""
        F = np.array([
            [1.0, 0.0, dt_hours, 0.0],
            [0.0, 1.0, 0.0, dt_hours],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q
        return float(self.state[0]), float(self.state[1])

    def update(self, meas_lat: float, meas_lon: float) -> None:
        """Update state estimate with observed measurement."""
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        z = np.array([meas_lat, meas_lon])
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance


class EventTrack:
    """
    Represents an evolving temporal track of a meteorological extreme event.
    """

    def __init__(self, track_id: str, initial_event: Dict[str, Any], use_kalman: bool = False):
        self.track_id = track_id
        self.events: List[Dict[str, Any]] = [initial_event]
        self.is_active = True
        self.consecutive_missed_steps = 0
        self.use_kalman = use_kalman

        if self.use_kalman:
            self.kalman = KalmanMotionModel(
                init_lat=float(initial_event["centroid_lat"]),
                init_lon=float(initial_event["centroid_lon"]),
            )
        else:
            self.kalman = None

    @property
    def latest_event(self) -> Dict[str, Any]:
        return self.events[-1]

    @property
    def start_time(self) -> str:
        return str(self.events[0]["timestamp"])

    @property
    def end_time(self) -> str:
        return str(self.events[-1]["timestamp"])

    @property
    def duration_hours(self) -> float:
        t_start = pd.to_datetime(self.start_time)
        t_end = pd.to_datetime(self.end_time)
        return float((t_end - t_start).total_seconds() / 3600.0)

    @property
    def total_displacement_km(self) -> float:
        first = self.events[0]
        last = self.events[-1]
        return float(haversine_distance_km(
            first["centroid_lat"], first["centroid_lon"],
            last["centroid_lat"], last["centroid_lon"],
        ))

    @property
    def cumulative_distance_km(self) -> float:
        if len(self.events) < 2:
            return 0.0
        total_d = 0.0
        for i in range(len(self.events) - 1):
            e1 = self.events[i]
            e2 = self.events[i + 1]
            total_d += haversine_distance_km(
                e1["centroid_lat"], e1["centroid_lon"],
                e2["centroid_lat"], e2["centroid_lon"],
            )
        return float(total_d)

    @property
    def mean_speed_kmh(self) -> float:
        dur = self.duration_hours
        if dur <= 0.0:
            return 0.0
        return float(self.cumulative_distance_km / dur)

    @property
    def net_bearing_deg(self) -> float:
        if len(self.events) < 2:
            return 0.0
        first = self.events[0]
        last = self.events[-1]
        return calculate_bearing_deg(
            first["centroid_lat"], first["centroid_lon"],
            last["centroid_lat"], last["centroid_lon"],
        )

    @property
    def max_intensity(self) -> float:
        return float(max(e.get("max_intensity", 0.0) for e in self.events))

    @property
    def max_area_km2(self) -> float:
        return float(max(e.get("area_km2", 0.0) for e in self.events))

    def add_event(self, event: Dict[str, Any]) -> None:
        """Append a newly matched event to the track."""
        self.events.append(event)
        self.consecutive_missed_steps = 0
        if self.kalman is not None:
            self.kalman.update(float(event["centroid_lat"]), float(event["centroid_lon"]))

    def mark_missed(self) -> None:
        """Mark a missed detection step for track lifecycle management."""
        self.consecutive_missed_steps += 1

    def get_track_state(self) -> Dict[str, Any]:
        """Get the latest spatial and intensity state of the track for matching."""
        state = dict(self.latest_event)
        if self.kalman is not None:
            pred_lat, pred_lon = self.kalman.predict(dt_hours=6.0)
            state["centroid_lat"] = pred_lat
            state["centroid_lon"] = pred_lon
        return state

    def to_summary_dict(self) -> Dict[str, Any]:
        """Generate a summary metrics dictionary for the completed track."""
        first = self.events[0]
        last = self.events[-1]
        max_int = max(float(e.get("max_intensity", 0.0)) for e in self.events)
        max_area = max(float(e.get("area_km2", 0.0)) for e in self.events)

        speeds = []
        for i in range(len(self.events) - 1):
            e1 = self.events[i]
            e2 = self.events[i + 1]
            d = haversine_distance_km(e1["centroid_lat"], e1["centroid_lon"], e2["centroid_lat"], e2["centroid_lon"])
            spd = calculate_speed_kmh(d, e1["timestamp"], e2["timestamp"])
            speeds.append(spd)
        max_speed = float(max(speeds)) if speeds else 0.0

        return {
            "track_id": self.track_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": round(self.duration_hours, 2),
            "num_events": len(self.events),
            "start_lat": round(float(first["centroid_lat"]), 4),
            "start_lon": round(float(first["centroid_lon"]), 4),
            "end_lat": round(float(last["centroid_lat"]), 4),
            "end_lon": round(float(last["centroid_lon"]), 4),
            "total_displacement_km": round(self.total_displacement_km, 2),
            "cumulative_distance_km": round(self.cumulative_distance_km, 2),
            "mean_speed_kmh": round(self.mean_speed_kmh, 2),
            "max_speed_kmh": round(max_speed, 2),
            "net_bearing_deg": round(self.net_bearing_deg, 1),
            "max_intensity": round(max_int, 2),
            "max_area_km2": round(max_area, 2),
        }


class BaselineEventTracker:
    """
    Classical baseline multi-object tracker for extreme weather anomaly objects across time steps.
    """

    def __init__(
        self,
        config: Optional[Union[EventAssociationConfig, Dict[str, Any]]] = None,
        use_kalman: bool = False,
    ):
        if isinstance(config, dict):
            self.config = EventAssociationConfig(**config)
        elif config is not None:
            self.config = config
        else:
            self.config = EventAssociationConfig()

        self.matcher = EventCandidateMatcher(self.config)
        self.use_kalman = use_kalman
        self.tracks: List[EventTrack] = []
        self._next_track_num = 1

    def _generate_track_id(self) -> str:
        tid = f"TRK_{self._next_track_num:04d}"
        self._next_track_num += 1
        return tid

    def track(self, df_events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute multi-timestep event tracking on a table of detected events.

        Args:
            df_events: DataFrame of detected events from Phase 2.

        Returns:
            Tuple[df_track_events, df_track_summary]:
                - df_track_events: Event rows annotated with 'track_id' and 'step_in_track'
                - df_track_summary: Per-track summary trajectory statistics
        """
        if df_events.empty:
            logger.warning("Empty events table provided; 0 tracks created.")
            return pd.DataFrame(), pd.DataFrame()

        # Sort events chronologically
        df_sorted = df_events.copy()
        df_sorted["dt_time"] = pd.to_datetime(df_sorted["timestamp"])
        df_sorted = df_sorted.sort_values("dt_time")

        timestamps = df_sorted["dt_time"].unique()
        logger.info(f"Tracking extreme events across {len(timestamps)} timestamps ({len(df_sorted)} total events)...")

        self.tracks = []
        self._next_track_num = 1

        active_tracks: List[EventTrack] = []

        for ts in timestamps:
            curr_events_df = df_sorted[df_sorted["dt_time"] == ts]
            curr_events: List[Dict[str, Any]] = curr_events_df.to_dict(orient="records")

            if not active_tracks:
                # Initialize first set of tracks
                for ev in curr_events:
                    new_trk = EventTrack(
                        track_id=self._generate_track_id(),
                        initial_event=ev,
                        use_kalman=self.use_kalman,
                    )
                    active_tracks.append(new_trk)
                    self.tracks.append(new_trk)
                continue

            # Extract latest state of active tracks
            track_states = [trk.get_track_state() for trk in active_tracks]

            # Solve data association
            matches, unmatched_track_idx, unmatched_event_idx = self.matcher.match_events(
                active_tracks=track_states,
                candidate_events=curr_events,
            )

            # 1. Update matched tracks
            for trk_idx, ev_idx, cost in matches:
                active_tracks[trk_idx].add_event(curr_events[ev_idx])

            # 2. Create new tracks for unmatched events
            for ev_idx in unmatched_event_idx:
                new_trk = EventTrack(
                    track_id=self._generate_track_id(),
                    initial_event=curr_events[ev_idx],
                    use_kalman=self.use_kalman,
                )
                active_tracks.append(new_trk)
                self.tracks.append(new_trk)

            # 3. Handle unmatched active tracks (lifecycle & termination)
            surviving_active_tracks: List[EventTrack] = []
            for trk_idx, trk in enumerate(active_tracks):
                if trk_idx in unmatched_track_idx:
                    trk.mark_missed()
                    if trk.consecutive_missed_steps <= self.config.max_missed_steps:
                        surviving_active_tracks.append(trk)
                    else:
                        trk.is_active = False
                else:
                    surviving_active_tracks.append(trk)

            active_tracks = surviving_active_tracks

        # Construct output dataframes
        track_event_rows: List[Dict[str, Any]] = []
        track_summary_rows: List[Dict[str, Any]] = []

        for trk in self.tracks:
            track_summary_rows.append(trk.to_summary_dict())
            for step_idx, ev in enumerate(trk.events):
                row = dict(ev)
                row["track_id"] = trk.track_id
                row["step_in_track"] = step_idx + 1
                track_event_rows.append(row)

        df_track_events = pd.DataFrame(track_event_rows)
        if "dt_time" in df_track_events.columns:
            df_track_events = df_track_events.drop(columns=["dt_time"])

        df_track_summary = pd.DataFrame(track_summary_rows)

        logger.info(f"Tracking complete: formed {len(df_track_summary)} tracks from {len(df_events)} events.")
        return df_track_events, df_track_summary

    @staticmethod
    def save_tracking_results(
        df_track_events: pd.DataFrame,
        df_track_summary: pd.DataFrame,
        output_dir: Union[str, Path],
        prefix: str = "precipitation",
        file_format: str = "parquet",
    ) -> Tuple[Path, Path]:
        """
        Save tracking output tables to disk.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        events_file = out_path / f"{prefix}_event_tracks.{file_format}"
        summary_file = out_path / f"{prefix}_track_summary.{file_format}"

        if file_format == "parquet":
            try:
                df_track_events.to_parquet(str(events_file), index=False)
                df_track_summary.to_parquet(str(summary_file), index=False)
                logger.info(f"Saved tracking results to Parquet: {events_file} and {summary_file}")
                return events_file, summary_file
            except Exception as e:
                logger.warning(f"Parquet engine exception ({e}); saving as CSV.")
                events_file = out_path / f"{prefix}_event_tracks.csv"
                summary_file = out_path / f"{prefix}_track_summary.csv"

        df_track_events.to_csv(str(events_file), index=False)
        df_track_summary.to_csv(str(summary_file), index=False)
        logger.info(f"Saved tracking results to CSV: {events_file} and {summary_file}")
        return events_file, summary_file
