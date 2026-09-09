from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.common import LaneObservation


@dataclass
class LaneBoundary:
    track_id: int
    points_image: List[List[float]]
    confidence: float
    visible: bool = True
    points_world: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "points_image": self.points_image,
            "points_world": self.points_world,
            "confidence": self.confidence,
            "visible": self.visible,
        }


class LaneTemporalTracker:
    def __init__(self, max_gap: int = 12, x_gate: float = 90.0):
        self.max_gap = max_gap
        self.x_gate = x_gate
        self.next_id = 1
        self.last_seen: Dict[int, int] = {}
        self.boundaries: Dict[int, LaneBoundary] = {}

    @staticmethod
    def _bottom_x(observation: LaneObservation) -> Optional[float]:
        if not observation.image_points:
            return None
        return float(observation.image_points[-1][0])

    def update(self, observations: Iterable[LaneObservation], frame_index: int) -> List[LaneBoundary]:
        observations = list(observations)
        current: List[LaneBoundary] = []
        used = set()
        candidates = []
        for observation in observations:
            x = self._bottom_x(observation)
            if x is None:
                continue
            for track_id, boundary in self.boundaries.items():
                if track_id in used or frame_index - self.last_seen.get(track_id, frame_index) > self.max_gap:
                    continue
                previous_x = boundary.points_image[-1][0] if boundary.points_image else x
                candidates.append((abs(x - previous_x), track_id, observation))
        for distance, track_id, observation in sorted(candidates, key=lambda item: item[0]):
            if distance > self.x_gate or track_id in used:
                continue
            boundary = LaneBoundary(track_id, observation.image_points, observation.confidence,
                                    points_world=observation.world_points)
            self.boundaries[track_id] = boundary
            self.last_seen[track_id] = frame_index
            current.append(boundary)
            used.add(track_id)
        for observation in observations:
            if observation.image_points and not any(boundary.points_image == observation.image_points for boundary in current):
                track_id = self.next_id
                self.next_id += 1
                boundary = LaneBoundary(track_id, observation.image_points, observation.confidence,
                                        points_world=observation.world_points)
                self.boundaries[track_id] = boundary
                self.last_seen[track_id] = frame_index
                current.append(boundary)
        return sorted(current, key=lambda boundary: boundary.points_image[-1][0] if boundary.points_image else 0)


@dataclass
class LaneGraph:
    nodes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    junctions: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, boundaries: List[LaneBoundary], frame_index: int) -> None:
        for boundary in boundaries:
            self.nodes[boundary.track_id] = {
                "track_id": boundary.track_id,
                "confidence": boundary.confidence,
                "last_frame": frame_index,
                "visible": boundary.visible,
            }
        ordered = sorted(boundaries, key=lambda item: item.points_image[-1][0] if item.points_image else 0)
        self.edges = [
            {"from": left.track_id, "to": right.track_id, "relation": "adjacent"}
            for left, right in zip(ordered, ordered[1:])
        ]
        if len(boundaries) >= 3:
            self.junctions.append({
                "frame_index": frame_index,
                "type": "junction_candidate",
                "incoming_boundaries": [boundary.track_id for boundary in boundaries],
                "confidence": float(np.mean([boundary.confidence for boundary in boundaries])),
                "reason": "multiple concurrent lane boundaries; requires temporal confirmation",
            })

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": self.edges, "junctions": self.junctions}


def validate_lane_graph(graph: LaneGraph) -> Dict[str, Any]:
    errors: List[str] = []
    node_ids = set(graph.nodes)
    for edge in graph.edges:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            errors.append("lane graph edge references an unknown boundary")
        if edge["from"] == edge["to"]:
            errors.append("lane graph contains a self-edge")
    return {"valid": not errors, "errors": errors, "junction_count": len(graph.junctions),
            "lane_boundary_count": len(graph.nodes)}
