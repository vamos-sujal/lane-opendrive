from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import scipy.special
import torch
from PIL import Image
from torchvision import transforms

from src.common import LaneObservation, MetricState
from src.detection.detector_base import BaseDetector


CULANE_ROW_ANCHORS = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class UFLDDetector(BaseDetector):
    """Official Ultra-Fast-Lane-Detection CULane checkpoint adapter.

    The model returns image-space lanes. It intentionally does not convert pixels
    to metric 3D coordinates.
    """

    name = "ufld_culane"

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda", repo_path: str = "/content/Ultra-Fast-Lane-Detection"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.repo_path = Path(repo_path)
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path or self.checkpoint_path
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError("UFLD checkpoint not found. Download the official CULane checkpoint first.")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"UFLD source repository not found: {self.repo_path}")
        sys.path.insert(0, str(self.repo_path))
        from model.model import parsingNet

        self.model = parsingNet(pretrained=False, backbone="18", cls_dim=(201, 18, 4), use_aux=False)
        # Explicitly retain the checkpoint's tensor/state-dict behavior across
        # PyTorch versions whose default ``weights_only`` setting changed.
        try:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(self.checkpoint_path, map_location="cpu")
        state = state.get("model", state)
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def warmup(self) -> None:
        if self.model is None:
            raise RuntimeError("UFLDDetector is not initialized.")
        with torch.inference_mode():
            self.model(torch.zeros((1, 3, 288, 800), device=self.device))

    def detect(self, frame: Any, frame_index: int, timestamp: float) -> List[LaneObservation]:
        if self.model is None:
            raise RuntimeError("UFLDDetector is not initialized.")
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)[0].detach().float().cpu().numpy()
        output = output[:, ::-1, :]
        probabilities = scipy.special.softmax(output[:-1], axis=0)
        positions = np.sum(probabilities * (np.arange(200) + 1).reshape(-1, 1, 1), axis=0)
        argmax = np.argmax(output, axis=0)
        positions[argmax == 200] = 0
        observations: List[LaneObservation] = []
        for lane_index in range(positions.shape[1]):
            points = []
            for row, x in enumerate(positions[:, lane_index]):
                if x > 0:
                    px = float(x * (width / 800.0))
                    py = float(CULANE_ROW_ANCHORS[17 - row] * (height / 288.0))
                    points.append([px, py])
            if len(points) < 3:
                continue
            observations.append(LaneObservation(
                detection_id=f"ufld_{frame_index}_{lane_index}", frame_index=frame_index, timestamp=timestamp,
                confidence=min(1.0, len(points) / 18.0), visibility=min(1.0, len(points) / 18.0),
                lane_marking="UNKNOWN", image_points=points, world_points=[], points_3d=[], uncertainty=1.0,
                metric_validity=MetricState.NON_METRIC,
                metadata={"model": self.name, "source": "official_culane_checkpoint"},
            ))
        return observations

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "status": "loaded" if self.model is not None else "unloaded", "metric_output": False}