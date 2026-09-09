from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from src.common import LaneTrack


@dataclass
class LaneGraph:
    nodes: Dict[int, Dict[str, object]] = field(default_factory=dict)
    edges: List[Dict[str, object]] = field(default_factory=list)
    events: List[Dict[str, object]] = field(default_factory=list)

    def add_tracks(self, tracks: Iterable[LaneTrack]) -> None:
        visible = []
        for track in tracks:
            if track.state.value == "terminated":
                continue
            self.nodes[track.lane_id] = {"lane_id": track.lane_id, "state": track.state.value,
                                        "confidence": track.confidence, "observations": len(track.observations)}
            if track.visible and track.feature_vector.get("end_x") is not None:
                visible.append(track)
        visible.sort(key=lambda item: item.feature_vector.get("end_x", 0.0))
        self.edges = [{"from": left.lane_id, "to": right.lane_id, "relation": "adjacent"}
                      for left, right in zip(visible, visible[1:])]

    def to_dict(self) -> Dict[str, object]:
        return {"nodes": list(self.nodes.values()), "edges": self.edges, "events": self.events}