from __future__ import annotations

from typing import Any, List

from src.common import LaneObservation
from src.detection.ufld import UFLDDetector


class ProductionLaneDetector:
    """Real pretrained lane-boundary adapter.

    UFLD is retained as a lane-boundary model only when its official CULane
    checkpoint loads successfully. Results remain domain-dependent and are
    validated before topology or OpenDRIVE export.
    """

    name = "ufld_culane_lane_boundary"

    def __init__(self, checkpoint_path: str, repo_path: str = "/content/Ultra-Fast-Lane-Detection"):
        self.detector = UFLDDetector(checkpoint_path=checkpoint_path, repo_path=repo_path)

    def load(self) -> None:
        self.detector.load_weights()
        self.detector.warmup()

    def detect(self, frame: Any, frame_index: int, timestamp: float) -> List[LaneObservation]:
        return self.detector.detect(frame, frame_index, timestamp)

    def status(self) -> dict:
        return {"name": self.name, "status": "loaded", "checkpoint": self.detector.checkpoint_path}
