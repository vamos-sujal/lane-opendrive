from __future__ import annotations

from math import hypot
from typing import Dict, List, Optional, Tuple

from src.common import LaneObservation, LaneTrack, TrackState


class LaneTracker:
    """Deterministic lane tracker with persistent IDs across temporal gaps."""

    def __init__(self, max_gap: int = 8, distance_gate: float = 8.0):
        self.tracks: Dict[int, LaneTrack] = {}
        self.next_lane_id = 1
        self.max_gap = max_gap
        self.distance_gate = distance_gate

    def _new_lane_id(self) -> int:
        lane_id = self.next_lane_id
        self.next_lane_id += 1
        return lane_id

    def update(self, observations: List[LaneObservation], frame_index: int) -> List[LaneTrack]:
        candidates: List[Tuple[float, int, int]] = []
        for lane_id, track in self.tracks.items():
            if track.last_frame_index is None or frame_index - track.last_frame_index > self.max_gap:
                continue
            for index, observation in enumerate(observations):
                distance = self._distance(track, observation)
                if distance <= self.distance_gate:
                    candidates.append((distance, lane_id, index))
        matched_tracks = set()
        matched_observations = set()
        for _, lane_id, index in sorted(candidates):
            if lane_id in matched_tracks or index in matched_observations:
                continue
            track = self.tracks[lane_id]
            observation = observations[index]
            track.state = TrackState.RECOVERED if not track.visible else TrackState.CONFIRMED
            track.observations.append(observation)
            track.last_frame_index = frame_index
            track.visible = True
            track.confidence = max(track.confidence, observation.confidence)
            self._update_features(track, observation)
            matched_tracks.add(lane_id)
            matched_observations.add(index)

        for index, observation in enumerate(observations):
            if index in matched_observations:
                continue
            lane_id = self._new_lane_id()
            track = LaneTrack(lane_id=lane_id, observations=[observation], last_frame_index=frame_index,
                              visible=True, confidence=observation.confidence)
            self._update_features(track, observation)
            self.tracks[lane_id] = track

        for lane_id, track in self.tracks.items():
            if lane_id not in matched_tracks and track.last_frame_index != frame_index:
                gap = frame_index - (track.last_frame_index or frame_index)
                track.visible = False
                if gap > self.max_gap:
                    track.state = TrackState.TERMINATED
                else:
                    track.state = TrackState.TEMPORARILY_LOST

        return list(self.tracks.values())

    @staticmethod
    def _endpoint(observation: LaneObservation) -> Optional[Tuple[float, float]]:
        points = observation.world_points or observation.points_3d or observation.image_points
        if not points:
            return None
        point = points[-1]
        return float(point[0]), float(point[1])

    def _distance(self, track: LaneTrack, observation: LaneObservation) -> float:
        current = self._endpoint(observation)
        previous = track.feature_vector.get("end_x"), track.feature_vector.get("end_y")
        if current is None or previous[0] is None or previous[1] is None:
            return self.distance_gate
        return hypot(current[0] - previous[0], current[1] - previous[1])

    def _update_features(self, track: LaneTrack, observation: LaneObservation) -> None:
        endpoint = self._endpoint(observation)
        if endpoint is not None:
            track.feature_vector.update(end_x=endpoint[0], end_y=endpoint[1])

    def persist(self) -> Dict[str, object]:
        return {
            "tracks": [
                {
                    "lane_id": track.lane_id,
                    "state": track.state.value,
                    "last_frame_index": track.last_frame_index,
                    "visible": track.visible,
                    "confidence": track.confidence,
                    "observations": len(track.observations),
                }
                for track in self.tracks.values()
            ]
        }
