from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common import LaneObservation
from src.detection.detector_base import BaseDetector


class LATRDetector(BaseDetector):
    """Reference implementation for a verified official LATR-style detector contract.

    This project intentionally does not silently invent metric geometry. The detector
    exposes the official model outputs in a calibration-neutral representation and
    leaves metric validity gating to the pipeline.
    """

    name = "latr"

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.initialized = False

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path or self.checkpoint_path
        if self.checkpoint_path is None:
            raise FileNotFoundError(
                "No LATR checkpoint path provided. Download weights with scripts/download_weights.py first."
            )
        raise RuntimeError(
            "The official LATR/MMDetection legacy runtime and adapter are not installed in this environment. "
            "The checkpoint file alone is not sufficient for inference; refusing to return fabricated detections."
        )

    def warmup(self) -> None:
        if not self.initialized:
            raise RuntimeError("LATRDetector is not initialized. Call load_weights() first.")

    def detect(self, frame: Any, frame_index: int, timestamp: float) -> List[LaneObservation]:
        if not self.initialized:
            raise RuntimeError("LATRDetector is not initialized.")
        raise RuntimeError(
            "LATR inference is unavailable because the official legacy runtime/adapter is not installed. "
            "No dummy or synthetic lane observations are produced."
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": "unavailable_without_official_runtime",
            "note": "The checkpoint is genuine, but this repository refuses to masquerade as LATR without the official legacy runtime and adapter.",
        }
