from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from src.common import LaneObservation, LaneTrack


def _writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height))


class VideoVisualizations:
    def __init__(self, output_dir: str, width: int, height: int, fps: float):
        root = Path(output_dir)
        self.output_dir = root
        self.original = _writer(root / "original_video.mp4", width, height, fps)
        self.overlay = _writer(root / "lane_overlay.mp4", width, height, fps)
        self.topology = _writer(root / "topology_overlay.mp4", width, height, fps)
        self.top_down: Optional[cv2.VideoWriter] = None
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_number = 0

    def enable_top_down(self, output_dir: str) -> None:
        self.top_down = _writer(Path(output_dir) / "top_down.mp4", 800, 600, self.fps)

    def write(self, frame: np.ndarray, observations: Iterable[LaneObservation], tracks: Iterable[LaneTrack], status: str) -> None:
        self.original.write(frame)
        overlay = frame.copy()
        topology = frame.copy()
        for obs in observations:
            points = np.asarray(obs.image_points, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2:
                continue
            points = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], False, (0, 220, 0), 3)
            cv2.polylines(topology, [points], False, (0, 220, 0), 3)
        for track in tracks:
            if not track.visible or not track.observations:
                continue
            obs = track.observations[-1]
            if obs.image_points:
                point = tuple(np.asarray(obs.image_points[-1], dtype=int))
                cv2.putText(overlay, f"lane {track.lane_id}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(topology, f"L{track.lane_id} {track.state.value}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
        cv2.putText(overlay, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(topology, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.overlay.write(overlay)
        self.topology.write(topology)
        if self.frame_number == 0 or self.frame_number % 30 == 0:
            cv2.imwrite(str(self.output_dir / f"lane_overlay_frame_{self.frame_number:06d}.jpg"), overlay)
            cv2.imwrite(str(self.output_dir / f"topology_overlay_frame_{self.frame_number:06d}.jpg"), topology)
        if self.top_down is not None:
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(canvas, "Top-down unavailable: metric geometry is not valid", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.top_down.write(canvas)
        self.frame_number += 1

    def close(self) -> None:
        for writer in (self.original, self.overlay, self.topology, self.top_down):
            if writer is not None:
                writer.release()