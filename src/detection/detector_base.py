from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.common import LaneObservation


class BaseDetector(ABC):
    """Abstract lane detector interface."""

    name: str = "base_detector"

    @abstractmethod
    def detect(self, frame: Any, frame_index: int, timestamp: float) -> List[LaneObservation]:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "status": "abstract"}
