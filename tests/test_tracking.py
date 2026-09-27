"""
Unit tests for SpatioAI Phase 3 Temporal Extreme Weather Event Tracking.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from src.events.association import EventAssociationConfig, EventCandidateMatcher
from src.events.tracking import BaselineEventTracker, EventTrack
from src.utils.geodesics import (
    calculate_bearing_deg,
    calculate_bbox_iou,
    calculate_speed_kmh,
    haversine_distance_km,
)


def test_geodesic_distance():
    """Test Haversine distance on known coordinate pairs."""
    # 1 degree on Equator is approximately 111.19 - 111.32 km
    dist_1deg_eq = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
    assert 111.0 < dist_1deg_eq < 111.5

    # Same point distance should be 0.0
    dist_same = haversine_distance_km(20.0, 85.0, 20.0, 85.0)
    assert np.isclose(dist_same, 0.0, atol=1e-5)

    # Known pair: Bhubaneswar (20.2961 N, 85.8245 E) to Kolkata (22.5726 N, 88.3639 E) ~ 365 km
    dist_bbsr_ccu = haversine_distance_km(20.2961, 85.8245, 22.5726, 88.3639)
    assert 350.0 < dist_bbsr_ccu < 380.0


def test_bearing_and_speed():
    """Test compass bearing angles and speed calculations."""
    # Due North: bearing = 0 deg
    bearing_n = calculate_bearing_deg(10.0, 80.0, 15.0, 80.0)
    assert np.isclose(bearing_n, 0.0, atol=1e-2)

    # Due East: bearing = 90 deg
    bearing_e = calculate_bearing_deg(0.0, 80.0, 0.0, 85.0)
    assert np.isclose(bearing_e, 90.0, atol=1e-2)

    # Speed: 120 km over 6 hours = 20 km/h
    speed = calculate_speed_kmh(120.0, "2024-01-01 00:00:00", "2024-01-01 06:00:00")
    assert np.isclose(speed, 20.0, atol=1e-3)


def test_bbox_iou():
    """Test bounding box Intersection-over-Union."""
    # Identical bounding boxes: IoU = 1.0
    box1 = (10.0, 12.0, 80.0, 82.0)
    assert np.isclose(calculate_bbox_iou(box1, box1), 1.0)

    # Disjoint boxes: IoU = 0.0
    box_disjoint = (20.0, 22.0, 90.0, 92.0)
    assert np.isclose(calculate_bbox_iou(box1, box_disjoint), 0.0)

    # Half-overlap box
    box_half = (11.0, 13.0, 80.0, 82.0)
    iou = calculate_bbox_iou(box1, box_half)
    assert 0.25 < iou < 0.40


def test_candidate_matching():
    """Test association cost computation and gating constraints."""
    config = EventAssociationConfig(
        max_centroid_distance_km=400.0,
        max_speed_kmh=100.0,
        max_area_change_ratio=3.0,
    )
    matcher = EventCandidateMatcher(config)

    track_state = {
        "centroid_lat": 20.0,
        "centroid_lon": 85.0,
        "timestamp": "2024-01-01 00:00:00",
        "min_lat": 19.0, "max_lat": 21.0,
        "min_lon": 84.0, "max_lon": 86.0,
        "area_km2": 10000.0,
        "max_intensity": 100.0,
    }

    # Plausible candidate event (displaced by ~60 km over 6h = 10 km/h)
    candidate_valid = {
        "centroid_lat": 20.4,
        "centroid_lon": 85.3,
        "timestamp": "2024-01-01 06:00:00",
        "min_lat": 19.4, "max_lat": 21.4,
        "min_lon": 84.3, "max_lon": 86.3,
        "area_km2": 11000.0,
        "max_intensity": 105.0,
    }
    cost, is_valid = matcher.compute_pair_cost(track_state, candidate_valid)
    assert is_valid
    assert 0.0 < cost < 0.5

    # Implausible candidate (too far: > 600 km away)
    candidate_far = dict(candidate_valid)
    candidate_far["centroid_lat"] = 28.0
    candidate_far["centroid_lon"] = 95.0
    cost_far, is_valid_far = matcher.compute_pair_cost(track_state, candidate_far)
    assert not is_valid_far
    assert cost_far >= 1e5


def test_event_association():
    """Test global Hungarian assignment vs Greedy assignment on 2 active tracks and 2 events."""
    config = EventAssociationConfig(algorithm="hungarian")
    matcher = EventCandidateMatcher(config)

    tracks = [
        {"centroid_lat": 18.0, "centroid_lon": 85.0, "timestamp": "2024-01-01 00:00:00",
         "min_lat": 17.0, "max_lat": 19.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 5000.0, "max_intensity": 80.0},
        {"centroid_lat": 12.0, "centroid_lon": 75.0, "timestamp": "2024-01-01 00:00:00",
         "min_lat": 11.0, "max_lat": 13.0, "min_lon": 74.0, "max_lon": 76.0, "area_km2": 8000.0, "max_intensity": 90.0},
    ]

    events = [
        # Closest to track 0
        {"event_id": "EV1", "centroid_lat": 18.3, "centroid_lon": 85.2, "timestamp": "2024-01-01 06:00:00",
         "min_lat": 17.3, "max_lat": 19.3, "min_lon": 84.2, "max_lon": 86.2, "area_km2": 5200.0, "max_intensity": 82.0},
        # Closest to track 1
        {"event_id": "EV2", "centroid_lat": 12.2, "centroid_lon": 75.3, "timestamp": "2024-01-01 06:00:00",
         "min_lat": 11.2, "max_lat": 13.2, "min_lon": 74.3, "max_lon": 76.3, "area_km2": 8100.0, "max_intensity": 92.0},
    ]

    matches, unmatched_trks, unmatched_evs = matcher.match_events(tracks, events)
    assert len(matches) == 2
    assert len(unmatched_trks) == 0
    assert len(unmatched_evs) == 0

    # Track 0 matched to EV1 (event idx 0), Track 1 matched to EV2 (event idx 1)
    match_dict = {trk_idx: ev_idx for trk_idx, ev_idx, _ in matches}
    assert match_dict[0] == 0
    assert match_dict[1] == 1


def test_track_creation_and_update():
    """Test creating a track and updating it with sequential observations."""
    initial_event = {
        "event_id": "EV_01",
        "timestamp": "2024-01-01 00:00:00",
        "centroid_lat": 20.0,
        "centroid_lon": 85.0,
        "area_km2": 10000.0,
        "max_intensity": 100.0,
    }
    track = EventTrack(track_id="TRK_0001", initial_event=initial_event)
    assert track.track_id == "TRK_0001"
    assert len(track.events) == 1
    assert track.is_active

    next_event = {
        "event_id": "EV_02",
        "timestamp": "2024-01-01 06:00:00",
        "centroid_lat": 20.5,
        "centroid_lon": 85.5,
        "area_km2": 12000.0,
        "max_intensity": 120.0,
    }
    track.add_event(next_event)
    assert len(track.events) == 2
    assert track.duration_hours == 6.0
    assert track.total_displacement_km > 50.0
    assert track.max_intensity == 120.0


def test_track_termination():
    """Test track lifecycle and termination after max_missed_steps."""
    config = EventAssociationConfig(max_missed_steps=1)
    tracker = BaselineEventTracker(config=config)

    # Event at t0, t1, then none at t2, t3
    df = pd.DataFrame([
        {"event_id": "E1", "timestamp": "2024-01-01 00:00:00", "centroid_lat": 20.0, "centroid_lon": 85.0,
         "min_lat": 19.0, "max_lat": 21.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 1000.0, "max_intensity": 50.0},
        {"event_id": "E2", "timestamp": "2024-01-01 06:00:00", "centroid_lat": 20.2, "centroid_lon": 85.2,
         "min_lat": 19.2, "max_lat": 21.2, "min_lon": 84.2, "max_lon": 86.2, "area_km2": 1000.0, "max_intensity": 55.0},
        # Distant unrelated event at t2 that should start a new track
        {"event_id": "E3", "timestamp": "2024-01-01 12:00:00", "centroid_lat": 10.0, "centroid_lon": 70.0,
         "min_lat": 9.0, "max_lat": 11.0, "min_lon": 69.0, "max_lon": 71.0, "area_km2": 1000.0, "max_intensity": 60.0},
        {"event_id": "E4", "timestamp": "2024-01-01 18:00:00", "centroid_lat": 10.2, "centroid_lon": 70.2,
         "min_lat": 9.2, "max_lat": 11.2, "min_lon": 69.2, "max_lon": 71.2, "area_km2": 1000.0, "max_intensity": 65.0},
    ])

    df_events, df_summary = tracker.track(df)
    assert len(df_summary) == 2
    # Track 1 has 2 events (terminated when unmatched), Track 2 has 2 events
    assert set(df_summary["num_events"].values) == {2}


def test_case1_single_moving_event():
    """Case 1: Single moving weather event over 4 time steps -> 1 continuous track."""
    tracker = BaselineEventTracker()
    df = pd.DataFrame([
        {"event_id": f"E{i}", "timestamp": f"2024-01-01 {i*6:02d}:00:00",
         "centroid_lat": 18.0 + i * 0.4, "centroid_lon": 88.0 - i * 0.4,
         "min_lat": 17.0 + i * 0.4, "max_lat": 19.0 + i * 0.4,
         "min_lon": 87.0 - i * 0.4, "max_lon": 89.0 - i * 0.4,
         "area_km2": 10000.0, "max_intensity": 100.0 + i * 5.0}
        for i in range(4)
    ])
    df_events, df_summary = tracker.track(df)
    assert len(df_summary) == 1
    assert df_summary.iloc[0]["num_events"] == 4
    assert df_summary.iloc[0]["duration_hours"] == 18.0


def test_case2_two_independent_systems():
    """Case 2: Two concurrent independent systems -> exactly 2 distinct tracks."""
    tracker = BaselineEventTracker()
    events = []
    for i in range(3):
        # System A (Bay of Bengal)
        events.append({
            "event_id": f"EA_{i}", "timestamp": f"2024-01-01 {i*6:02d}:00:00",
            "centroid_lat": 19.0 + i * 0.3, "centroid_lon": 86.0 + i * 0.2,
            "min_lat": 18.0 + i * 0.3, "max_lat": 20.0 + i * 0.3,
            "min_lon": 85.0 + i * 0.2, "max_lon": 87.0 + i * 0.2,
            "area_km2": 20000.0, "max_intensity": 120.0,
        })
        # System B (Arabian Sea)
        events.append({
            "event_id": f"EB_{i}", "timestamp": f"2024-01-01 {i*6:02d}:00:00",
            "centroid_lat": 10.0 + i * 0.2, "centroid_lon": 74.0 + i * 0.2,
            "min_lat": 9.0 + i * 0.2, "max_lat": 11.0 + i * 0.2,
            "min_lon": 73.0 + i * 0.2, "max_lon": 75.0 + i * 0.2,
            "area_km2": 15000.0, "max_intensity": 90.0,
        })

    df = pd.DataFrame(events)
    df_events, df_summary = tracker.track(df)
    assert len(df_summary) == 2
    assert list(df_summary["num_events"].values) == [3, 3]


def test_case3_temporary_missing_detection():
    """Case 3: Temporary missing detection bridged by max_missed_steps >= 1."""
    config = EventAssociationConfig(max_missed_steps=1, max_centroid_distance_km=300.0)
    tracker = BaselineEventTracker(config=config)

    # Event at t0 (00h), t1 (06h), missing at t2 (12h), reappears at t3 (18h)
    df = pd.DataFrame([
        {"event_id": "E1", "timestamp": "2024-01-01 00:00:00", "centroid_lat": 20.0, "centroid_lon": 85.0,
         "min_lat": 19.0, "max_lat": 21.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 10000.0, "max_intensity": 100.0},
        {"event_id": "E2", "timestamp": "2024-01-01 06:00:00", "centroid_lat": 20.3, "centroid_lon": 85.3,
         "min_lat": 19.3, "max_lat": 21.3, "min_lon": 84.3, "max_lon": 86.3, "area_km2": 10000.0, "max_intensity": 105.0},
        # Gap at 12:00:00 (no event)
        {"event_id": "E3", "timestamp": "2024-01-01 18:00:00", "centroid_lat": 20.8, "centroid_lon": 85.8,
         "min_lat": 19.8, "max_lat": 21.8, "min_lon": 84.8, "max_lon": 86.8, "area_km2": 10000.0, "max_intensity": 110.0},
    ])

    df_events, df_summary = tracker.track(df)
    assert len(df_summary) == 1
    assert df_summary.iloc[0]["num_events"] == 3
    assert df_summary.iloc[0]["duration_hours"] == 18.0


def test_case4_distant_events_not_merged():
    """Case 4: Events separated beyond max_centroid_distance_km are not merged."""
    config = EventAssociationConfig(max_centroid_distance_km=200.0)
    tracker = BaselineEventTracker(config=config)

    df = pd.DataFrame([
        {"event_id": "E1", "timestamp": "2024-01-01 00:00:00", "centroid_lat": 15.0, "centroid_lon": 75.0,
         "min_lat": 14.0, "max_lat": 16.0, "min_lon": 74.0, "max_lon": 76.0, "area_km2": 5000.0, "max_intensity": 60.0},
        # 800 km away in next step
        {"event_id": "E2", "timestamp": "2024-01-01 06:00:00", "centroid_lat": 22.0, "centroid_lon": 85.0,
         "min_lat": 21.0, "max_lat": 23.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 5000.0, "max_intensity": 65.0},
    ])

    df_events, df_summary = tracker.track(df)
    assert len(df_summary) == 2


def test_track_summary_metrics():
    """Test full calculation of trajectory metrics in track summary table."""
    tracker = BaselineEventTracker()
    df = pd.DataFrame([
        {"event_id": "E1", "timestamp": "2024-01-01 00:00:00", "centroid_lat": 18.0, "centroid_lon": 85.0,
         "min_lat": 17.0, "max_lat": 19.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 10000.0, "max_intensity": 100.0},
        {"event_id": "E2", "timestamp": "2024-01-01 12:00:00", "centroid_lat": 20.0, "centroid_lon": 85.0,
         "min_lat": 19.0, "max_lat": 21.0, "min_lon": 84.0, "max_lon": 86.0, "area_km2": 15000.0, "max_intensity": 130.0},
    ])

    _, df_summary = tracker.track(df)
    s = df_summary.iloc[0]
    assert s["track_id"] == "TRK_0001"
    assert s["num_events"] == 2
    assert s["duration_hours"] == 12.0
    assert s["start_lat"] == 18.0
    assert s["end_lat"] == 20.0
    assert s["max_intensity"] == 130.0
    assert s["max_area_km2"] == 15000.0
    assert np.isclose(s["net_bearing_deg"], 0.0, atol=1.0)  # Due North
    assert s["total_displacement_km"] > 200.0
