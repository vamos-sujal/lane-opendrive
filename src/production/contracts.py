from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MeasurementStatus(str, Enum):
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


@dataclass
class Measurement:
    value: Optional[float]
    unit: str
    status: MeasurementStatus
    uncertainty: Optional[float] = None
    source: str = "unknown"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class ModelStatus:
    name: str
    status: str
    version: Optional[str] = None
    checkpoint: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    lanes: List[Dict[str, Any]] = field(default_factory=list)
    vehicles: List[Dict[str, Any]] = field(default_factory=list)
    junctions: List[Dict[str, Any]] = field(default_factory=list)
    ego_speed: Measurement = field(default_factory=lambda: Measurement(
        None, "m/s", MeasurementStatus.UNAVAILABLE,
        reason="scale_not_observable",
    ))
    nearest_vehicle_distance: Measurement = field(default_factory=lambda: Measurement(
        None, "m", MeasurementStatus.UNAVAILABLE,
        reason="metric_depth_not_validated",
    ))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ego_speed"] = self.ego_speed.to_dict()
        payload["nearest_vehicle_distance"] = self.nearest_vehicle_distance.to_dict()
        return payload


@dataclass
class RunContract:
    input_path: str
    frame_count: int
    fps: float
    duration_s: float
    calibration_status: str
    scale_source: str
    models: List[ModelStatus] = field(default_factory=list)
    frames: List[FrameResult] = field(default_factory=list)
    road_length: Measurement = field(default_factory=lambda: Measurement(
        None, "m", MeasurementStatus.UNAVAILABLE,
        reason="road_extent_and_metric_scale_not_validated",
    ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "calibration_status": self.calibration_status,
            "scale_source": self.scale_source,
            "models": [model.to_dict() for model in self.models],
            "frames": [frame.to_dict() for frame in self.frames],
            "road_length": self.road_length.to_dict(),
        }
