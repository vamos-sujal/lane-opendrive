from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.common import LaneObservation, MetricState


@dataclass(frozen=True)
class GroundPlaneCalibration:
    """Camera calibration for a z=0 ground plane in metres.

    World coordinates use x=right, y=forward, z=up. ``pitch_deg`` is the
    downward camera pitch. Distortion must already be removed from inputs.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float
    pitch_deg: float
    lane_width_m: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["GroundPlaneCalibration"]:
        if not config or not config.get("calibration_available", False):
            return None
        intrinsics = config.get("intrinsics") or {}
        required = ("fx", "fy", "cx", "cy", "height_m", "pitch_deg")
        values = {key: intrinsics.get(key, config.get(key)) for key in required}
        if any(value is None for value in values.values()):
            return None
        calibration = cls(**{key: float(value) for key, value in values.items()},
                          lane_width_m=float(config["lane_width_m"]) if config.get("lane_width_m") else None)
        if calibration.fx <= 0 or calibration.fy <= 0 or calibration.height_m <= 0:
            return None
        return calibration

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def world_to_camera(self) -> np.ndarray:
        pitch = np.deg2rad(self.pitch_deg)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -np.sin(pitch), -np.cos(pitch)],
            [0.0, np.cos(pitch), -np.sin(pitch)],
        ])


def image_to_ground(point: Iterable[float], calibration: GroundPlaneCalibration) -> Optional[List[float]]:
    """Intersect an image ray with the calibrated z=0 ground plane."""
    pixel = np.array([float(point[0]), float(point[1]), 1.0])
    ray_camera = np.linalg.inv(calibration.camera_matrix()) @ pixel
    ray_world = calibration.world_to_camera().T @ ray_camera
    if ray_world[2] >= -1e-9:
        return None
    scale = -calibration.height_m / ray_world[2]
    world = np.array([0.0, 0.0, calibration.height_m]) + scale * ray_world
    if not np.isfinite(world).all():
        return None
    return [float(world[0]), float(world[1]), 0.0]


def project_observation(observation: LaneObservation, calibration: Optional[GroundPlaneCalibration]) -> LaneObservation:
    if calibration is None or len(observation.image_points) < 3:
        observation.metric_validity = MetricState.NON_METRIC
        observation.world_points = []
        observation.points_3d = []
        return observation
    projected = [image_to_ground(point, calibration) for point in observation.image_points]
    valid_points = [point for point in projected if point is not None]
    if len(valid_points) < 3:
        observation.metric_validity = MetricState.INVALID
        observation.world_points = []
        observation.points_3d = []
        return observation
    observation.world_points = valid_points
    observation.points_3d = list(observation.world_points)
    observation.metric_validity = MetricState.METRIC_VALID
    observation.uncertainty = 0.0
    return observation


def estimate_ground_speed(previous: List[Tuple[float, float]], current: List[Tuple[float, float]], dt: float) -> Optional[float]:
    """Estimate camera/road-relative speed from tracked static lane points."""
    if dt <= 0 or len(previous) < 2 or len(current) < 2:
        return None
    count = min(len(previous), len(current))
    distances = [float(np.linalg.norm(np.subtract(current[index], previous[index]))) for index in range(count)]
    distances = [value for value in distances if np.isfinite(value)]
    if not distances:
        return None
    return float(np.median(distances) / dt)


def calibration_summary(calibration: Optional[GroundPlaneCalibration]) -> Dict[str, Any]:
    if calibration is None:
        return {"available": False, "reason": "trusted_intrinsics_height_and_pitch_required"}
    return {"available": True, "height_m": calibration.height_m, "pitch_deg": calibration.pitch_deg,
            "lane_width_m": calibration.lane_width_m, "distortion_model": "pre_undistorted_required"}