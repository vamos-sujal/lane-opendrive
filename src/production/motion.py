from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class VisualOdometryAdapter:
    """Monocular essential-matrix VO fused with model depth.

    The result is model-scale until camera intrinsics and physical scale are
    independently validated.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_depth: Optional[np.ndarray] = None
        self.distance_m = 0.0
        self.last_speed_mps: Optional[float] = None
        self.samples = 0
        self.pose_failures = 0
        focal = float(max(width, height))
        self.camera_matrix = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _static_points(self, gray: np.ndarray, road_mask: np.ndarray,
                       vehicles: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        mask = np.where(road_mask, 255, 0).astype(np.uint8)
        for vehicle in vehicles:
            x1, y1, x2, y2 = [int(value) for value in vehicle.get("bbox_xyxy", (0, 0, 0, 0))]
            cv2.rectangle(mask, (max(0, x1 - 12), max(0, y1 - 12)),
                          (min(self.width - 1, x2 + 12), min(self.height - 1, y2 + 12)), 0, -1)
        return cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01,
                                       minDistance=8, mask=mask)

    def update(self, frame: np.ndarray, depth: np.ndarray, road_mask: np.ndarray,
               vehicles: List[Dict[str, Any]], dt: float) -> Optional[float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = self._static_points(gray, road_mask, vehicles)
        if self.previous_gray is None or points is None or len(points) < 12 or dt <= 0:
            self.previous_gray, self.previous_depth = gray, depth
            return None
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        old, new = points.reshape(-1, 2), next_points.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool)
        old, new = old[valid], new[valid]
        if len(old) < 12:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        essential, _ = cv2.findEssentialMat(old, new, self.camera_matrix,
                                            method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if essential is None:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        _, _, _, pose_mask = cv2.recoverPose(essential, old, new, self.camera_matrix)
        inlier_mask = pose_mask.reshape(-1).astype(bool)
        old_inliers, new_inliers = old[inlier_mask], new[inlier_mask]
        if len(old_inliers) < 8:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        depth_y = np.clip(old_inliers[:, 1].astype(int), 0, self.previous_depth.shape[0] - 1)
        depth_x = np.clip(old_inliers[:, 0].astype(int), 0, self.previous_depth.shape[1] - 1)
        depths = self.previous_depth[depth_y, depth_x]
        depths = depths[np.isfinite(depths) & (depths > 0)]
        if len(depths) < 8:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        flow = np.linalg.norm(new_inliers - old_inliers, axis=1)
        finite_flow = flow[np.isfinite(flow)]
        translation_m = float(np.median(depths) * np.median(finite_flow) / self.camera_matrix[0, 0])
        if not np.isfinite(translation_m) or translation_m <= 0 or translation_m > 50:
            self.pose_failures += 1
            self.previous_gray, self.previous_depth = gray, depth
            return None
        self.distance_m += translation_m
        self.last_speed_mps = translation_m / dt
        self.samples += 1
        self.previous_gray, self.previous_depth = gray, depth
        return self.last_speed_mps

    def report(self) -> Dict[str, Any]:
        status = "ESTIMATED_MODEL_SCALE" if self.samples else "UNAVAILABLE_NO_VALID_POSE"
        return {
            "motion_source": "essential_matrix_visual_odometry_plus_depth_fusion",
            "distance_travelled_m": self.distance_m if self.samples else None,
            "distance_travelled_status": status,
            "dashcam_speed_mps": self.last_speed_mps,
            "dashcam_speed_kmh": self.last_speed_mps * 3.6 if self.last_speed_mps is not None else None,
            "dashcam_speed_status": status,
            "road_length_m": self.distance_m if self.samples else None,
            "road_length_status": status,
            "uncertainty_note": "Monocular VO direction is fused with model depth; validate intrinsics and physical scale before calling this measured metric.",
            "valid_pose_samples": self.samples,
            "pose_failures": self.pose_failures,
        }
