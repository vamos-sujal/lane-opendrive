from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 0.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = (self.frame_count / self.fps) if self.fps > 0 else 0.0

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=self.frame_count,
            duration=self.duration,
            codec="",
        )

    def frames(self) -> Iterator[Tuple[int, float, object]]:
        frame_index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            timestamp = frame_index / self.fps if self.fps > 0 else 0.0
            yield frame_index, timestamp, frame
            frame_index += 1
        self.capture.release()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def resolve_input_video(path: str) -> str:
    if not path:
        raise ValueError("No input path provided.")
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        matches = []
        for name in sorted(os.listdir(path)):
            if name.lower().endswith((".mp4", ".mov", ".avi")):
                matches.append(os.path.join(path, name))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No video file found in input directory: {path}")
        if "input.mp4" in [os.path.basename(p) for p in matches]:
            return os.path.join(path, "input.mp4")
        raise FileNotFoundError(
            f"Multiple video files found in {path}. Please choose one explicitly."
        )
    raise FileNotFoundError(f"Input file or directory does not exist: {path}")


def write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
