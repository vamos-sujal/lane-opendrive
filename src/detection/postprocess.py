from __future__ import annotations

from typing import Any, Dict, List

from src.common import LaneObservation, MetricState


def sanitize_observation(obs: LaneObservation) -> LaneObservation:
    if obs.metric_validity not in {MetricState.METRIC_VALID, MetricState.METRIC_UNCERTAIN, MetricState.NON_METRIC, MetricState.INVALID}:
        obs.metric_validity = MetricState.METRIC_UNCERTAIN
    if not obs.points_3d:
        obs.points_3d = [[0.0, 0.0, 0.0]]
    if not obs.image_points:
        obs.image_points = [[0.0, 0.0]]
    if not obs.world_points:
        obs.world_points = [[0.0, 0.0, 0.0]]
    return obs


def coerce_observations(raw: List[Dict[str, Any]]) -> List[LaneObservation]:
    outs: List[LaneObservation] = []
    for item in raw:
        obs = LaneObservation(
            detection_id=item.get("detection_id", "obs"),
            frame_index=item.get("frame_index", 0),
            timestamp=item.get("timestamp", 0.0),
            points_3d=item.get("points_3d", [[0.0, 0.0, 0.0]]),
            confidence=float(item.get("confidence", 0.0)),
            visibility=float(item.get("visibility", 0.0)),
            lane_marking=item.get("lane_marking", "UNKNOWN"),
            image_points=item.get("image_points", [[0.0, 0.0]]),
            world_points=item.get("world_points", [[0.0, 0.0, 0.0]]),
            uncertainty=float(item.get("uncertainty", 1.0)),
            metric_validity=MetricState(item.get("metric_validity", MetricState.METRIC_UNCERTAIN.value)),
            metadata=item.get("metadata", {}),
        )
        outs.append(sanitize_observation(obs))
    return outs
