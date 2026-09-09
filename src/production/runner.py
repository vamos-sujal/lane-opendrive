from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

from src.io.video_reader import VideoReader
from src.production.contracts import FrameResult, Measurement, MeasurementStatus, ModelStatus, RunContract
from src.production.model_registry import LoadedModels, runtime_report
from src.production.motion import VisualOdometryAdapter
from src.production.lane_topology import LaneGraph, LaneTemporalTracker, validate_lane_graph
from src.production.export import export_validated_opendrive


CITYSCAPES_ROAD_ID = 0


def _model_status(models: LoadedModels) -> List[ModelStatus]:
    return [
        ModelStatus("ultralytics_vehicle_detector", "loaded", checkpoint="yolo11m.pt"),
        ModelStatus("depth_anything_v2_metric_outdoor", "loaded", checkpoint="Depth-Anything-V2-Metric-Outdoor-Base-hf"),
        ModelStatus("segformer_cityscapes", "loaded", checkpoint="segformer-b2-finetuned-cityscapes-1024-1024"),
        ModelStatus("lane_detector", "unavailable", error="No validated arbitrary-dashcam lane checkpoint configured."),
        ModelStatus("junction_topology", "unavailable", error="No validated arbitrary-dashcam topology checkpoint configured."),
    ]


def _depth_map(models: LoadedModels, frame: np.ndarray, device: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = models.depth_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        prediction = models.depth_model(**inputs).predicted_depth
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False
    ).squeeze().float().cpu().numpy()
    return prediction


def _scene_mask(models: LoadedModels, frame: np.ndarray, device: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = models.scene_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = models.scene_model(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze().cpu().numpy()


def _vehicles(models: LoadedModels, frame: np.ndarray, device: str) -> List[Dict[str, Any]]:
    results = models.object_model.track(frame, persist=True, device=device, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes.xyxy)
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.float().cpu().tolist()
    coordinates = boxes.xyxy.float().cpu().tolist()
    return [
        {"track_id": track_id, "class_id": class_id, "confidence": confidence,
         "bbox_xyxy": [float(value) for value in bbox]}
        for track_id, class_id, confidence, bbox in zip(ids, classes, confidences, coordinates)
    ]


def run_production(input_path: str, output_dir: str, models: LoadedModels, device: str = "cuda",
                   scene_stride: int = 5, lane_detector: Any = None) -> Dict[str, Any]:
    scene_stride = max(1, scene_stride)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(input_path)
    metadata = reader.metadata
    contract = RunContract(input_path, metadata.frame_count, metadata.fps, metadata.duration,
                           calibration_status="NOT_SUPPLIED", scale_source="unknown",
                           models=_model_status(models))
    start = time.perf_counter()
    motion = VisualOdometryAdapter(metadata.width, metadata.height)
    lane_tracker = LaneTemporalTracker()
    lane_graph = LaneGraph()
    previous_timestamp = None
    latest_lane_boundaries = []
    latest_scene = None
    for frame_index, timestamp, frame in reader.frames():
        vehicles = _vehicles(models, frame, device)
        lane_boundaries = []
        if lane_detector is not None:
            lane_observations = lane_detector.detect(frame, frame_index, timestamp)
            lane_boundaries = lane_tracker.update(lane_observations, frame_index)
            latest_lane_boundaries = lane_boundaries
            lane_graph.update(lane_boundaries, frame_index)
        if frame_index % scene_stride == 0 or latest_scene is None:
            latest_scene = _scene_mask(models, frame, device)
        depth = _depth_map(models, frame, device)
        for vehicle in vehicles:
            x1, y1, x2, y2 = [int(value) for value in vehicle["bbox_xyxy"]]
            center_x = max(0, min(depth.shape[1] - 1, (x1 + x2) // 2))
            center_y = max(0, min(depth.shape[0] - 1, (y1 + y2) // 2))
            vehicle["depth_model_value"] = float(depth[center_y, center_x])
            vehicle["distance"] = Measurement(
                float(depth[center_y, center_x]), "m", MeasurementStatus.ESTIMATED,
                uncertainty=None, source="Depth-Anything-V2-Metric-Outdoor-Base-hf",
                reason="camera calibration and scale not independently validated",
            ).to_dict()
        road_fraction = float(np.mean(latest_scene == CITYSCAPES_ROAD_ID)) if latest_scene is not None else 0.0
        road_mask = latest_scene == CITYSCAPES_ROAD_ID if latest_scene is not None else np.ones(depth.shape, dtype=bool)
        motion.update(
            frame,
            depth,
            road_mask,
            vehicles,
            timestamp - previous_timestamp if previous_timestamp is not None else 0.0,
        )
        previous_timestamp = timestamp
        estimated_distances = [
            vehicle["distance"]["value"]
            for vehicle in vehicles
            if vehicle.get("distance", {}).get("value") is not None
        ]
        nearest_vehicle_distance = Measurement(
            min(estimated_distances), "m", MeasurementStatus.ESTIMATED,
            uncertainty=None, source="Depth-Anything-V2-Metric-Outdoor-Base-hf",
            reason="model depth is not validated against this camera's scale",
        ) if estimated_distances else Measurement(
            None, "m", MeasurementStatus.UNAVAILABLE,
            source="depth_model", reason="no vehicle was detected",
        )
        speed_measurement = Measurement(
            motion.last_speed_mps, "m/s", MeasurementStatus.ESTIMATED,
            source="monocular_optical_flow_and_metric_depth",
            reason="camera scale and intrinsics not independently validated",
        ) if motion.last_speed_mps is not None else Measurement(
            None, "m/s", MeasurementStatus.UNAVAILABLE,
            source="monocular_optical_flow", reason="insufficient stable road features",
        )
        contract.frames.append(FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp,
            lanes=[boundary.to_dict() for boundary in lane_boundaries],
            vehicles=vehicles,
            junctions=[],
            ego_speed=speed_measurement,
            nearest_vehicle_distance=nearest_vehicle_distance,
        ))
        if frame_index % 100 == 0:
            print(f"Processed frame {frame_index}/{metadata.frame_count}; road_fraction={road_fraction:.3f}", flush=True)
    reader.close()
    payload = contract.to_dict()
    payload.update(motion.report())
    payload["lane_graph"] = lane_graph.to_dict()
    payload["lane_validation"] = validate_lane_graph(lane_graph)
    xodr_report = export_validated_opendrive(
        [boundary.to_dict() for boundary in latest_lane_boundaries],
        str(root / "output.xodr"),
        lane_width_m=None,
    )
    payload["opendrive"] = {
        "emitted": bool(xodr_report.get("valid")),
        "status": "VALIDATED" if xodr_report.get("valid") else "UNAVAILABLE_NO_METRIC_LANE_GEOMETRY",
        "validation": xodr_report,
    }
    payload["runtime"] = runtime_report()
    payload["processing_fps"] = len(contract.frames) / max(time.perf_counter() - start, 1e-6)
    payload["capabilities"] = {
        "vehicle_detection": True,
        "estimated_depth": True,
        "estimated_vehicle_distance": True,
        "estimated_distance_travelled": bool(motion.samples),
        "estimated_ego_speed": bool(motion.samples),
        "metric_distance": False,
        "ego_speed": False,
        "lanes": lane_detector is not None,
        "junctions": bool(lane_graph.junctions),
        "opendrive": False,
    }
    (root / "production_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
