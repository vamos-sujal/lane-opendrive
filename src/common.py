from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricState(str, Enum):
    METRIC_VALID = "METRIC_VALID"
    METRIC_UNCERTAIN = "METRIC_UNCERTAIN"
    NON_METRIC = "NON_METRIC"
    INVALID = "INVALID"


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TEMPORARILY_LOST = "temporarily_lost"
    RECOVERED = "recovered"
    TERMINATED = "terminated"


@dataclass
class LaneObservation:
    detection_id: str
    frame_index: int
    timestamp: float
    points_3d: List[List[float]] = field(default_factory=list)
    confidence: float = 0.0
    visibility: float = 0.0
    lane_marking: str = "UNKNOWN"
    image_points: List[List[float]] = field(default_factory=list)
    world_points: List[List[float]] = field(default_factory=list)
    uncertainty: float = 0.0
    metric_validity: MetricState = MetricState.METRIC_UNCERTAIN
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneTrack:
    lane_id: int
    state: TrackState = TrackState.TENTATIVE
    observations: List[LaneObservation] = field(default_factory=list)
    last_frame_index: Optional[int] = None
    visible: bool = False
    confidence: float = 0.0
    feature_vector: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricReport:
    status: MetricState = MetricState.METRIC_UNCERTAIN
    source: str = "unknown"
    calibration_source: str = "unknown"
    uncertainty: float = 1.0
    confidence: float = 0.0
    valid: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
