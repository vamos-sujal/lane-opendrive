from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.common import LaneTrack, MetricState


def fit_lane_geometry(track: LaneTrack, degree: int = 2) -> Dict[str, object]:
    points: List[List[float]] = []
    for observation in track.observations:
        points.extend(observation.world_points or observation.points_3d)
    if len(points) < degree + 1:
        return {"lane_id": track.lane_id, "status": MetricState.INVALID.value, "points": points, "coefficients": []}
    array = np.asarray(points, dtype=float)
    if not np.isfinite(array).all() or any(obs.metric_validity != MetricState.METRIC_VALID for obs in track.observations):
        return {"lane_id": track.lane_id, "status": MetricState.NON_METRIC.value, "points": points, "coefficients": []}
    order = np.argsort(array[:, 1])
    coefficients = np.polyfit(array[order, 1], array[order, 0], degree).tolist()
    return {"lane_id": track.lane_id, "status": MetricState.METRIC_VALID.value, "points": points, "coefficients": coefficients}


def fit_all_tracks(tracks: Iterable[LaneTrack]) -> List[Dict[str, object]]:
    return [fit_lane_geometry(track) for track in tracks if track.state.value != "terminated"]