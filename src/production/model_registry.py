from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


# T4 profile: these sizes leave VRAM for frame tensors and tracking. Larger
# checkpoints can be selected only after a real smoke test on the target GPU.
OBJECT_MODEL = "yolo11m.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
SCENE_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


@dataclass
class LoadedModels:
    object_model: Any
    depth_processor: Any
    depth_model: Any
    scene_processor: Any
    scene_model: Any


def _version(module_name: str) -> Optional[str]:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", None)


def runtime_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {"packages": {}, "cuda": False, "gpu": None}
    for name in ("torch", "torchvision", "cv2", "numpy", "transformers", "ultralytics"):
        try:
            report["packages"][name] = _version(name) or "imported"
        except Exception as exc:
            report["packages"][name] = f"ERROR: {type(exc).__name__}: {exc}"
    try:
        import torch
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return report


def load_models(device: str = "cuda") -> LoadedModels:
    """Load real pretrained models and fail before video processing on mismatch."""
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        AutoModelForSemanticSegmentation,
    )
    from ultralytics import YOLO

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production T4 runtime.")

    object_model = YOLO(OBJECT_MODEL)
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL)
    depth_model.to(device).eval()
    scene_processor = AutoImageProcessor.from_pretrained(SCENE_MODEL)
    scene_model = AutoModelForSemanticSegmentation.from_pretrained(SCENE_MODEL)
    scene_model.to(device).eval()
    return LoadedModels(object_model, depth_processor, depth_model, scene_processor, scene_model)
