from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.common import LaneObservation, MetricState
from src.detection.latr import LATRDetector
from src.detection.ufld import UFLDDetector
from src.detection.postprocess import sanitize_observation
from src.geometry.fitting import fit_all_tracks
from src.io.video_reader import VideoReader, write_json
from src.opendrive.validator import validate_xodr
from src.topology.graph import LaneGraph
from src.tracking.tracker import LaneTracker
from src.visualization import VideoVisualizations


def run_pipeline(input_path: str, output_dir: str, checkpoint: str | None = None, detector_name: str = "ufld_culane") -> Dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    reader = VideoReader(input_path)
    meta = reader.metadata
    visualizations = VideoVisualizations(str(root / "visualizations"), meta.width, meta.height, meta.fps)
    tracker = LaneTracker()
    detector = UFLDDetector(checkpoint_path=checkpoint) if detector_name == "ufld_culane" else LATRDetector(checkpoint_path=checkpoint)
    detector_error = None
    try:
        detector.load_weights()
        detector.warmup()
    except Exception as exc:
        detector_error = str(exc)
        logging.error("Detector initialization failed: %s", exc)

    all_observations: List[LaneObservation] = []
    frame_lane_counts: List[int] = []
    processed = 0
    start = time.perf_counter()
    try:
        for frame_index, timestamp, frame in reader.frames():
            observations: List[LaneObservation] = []
            if detector_error is None:
                try:
                    observations = [sanitize_observation(obs) for obs in detector.detect(frame, frame_index, timestamp)]
                except Exception as exc:
                    detector_error = str(exc)
                    logging.error("Detector failed at frame %s: %s", frame_index, exc)
            tracks = tracker.update(observations, frame_index)
            all_observations.extend(observations)
            frame_lane_counts.append(len(observations))
            visualizations.write(frame, observations, tracks, "LATR unavailable / fail-closed" if detector_error else "LATR inference")
            processed += 1
    finally:
        reader.close()
        visualizations.close()

    tracks = list(tracker.tracks.values())
    geometries = fit_all_tracks(tracks) if not detector_error else []
    graph = LaneGraph()
    graph.add_tracks(tracks)
    elapsed = time.perf_counter() - start
    metric_status = MetricState.NON_METRIC.value
    image_lane_lengths = []
    for track in tracks:
        latest = track.observations[-1] if track.observations else None
        if latest and len(latest.image_points) >= 2:
            points = latest.image_points
            length_px = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(points, points[1:]))
            image_lane_lengths.append({"lane_id": track.lane_id, "visible_length_px": length_px})
    relative_report = {
        "lane_count_min": min(frame_lane_counts) if frame_lane_counts else 0,
        "lane_count_max": max(frame_lane_counts) if frame_lane_counts else 0,
        "lane_count_median": sorted(frame_lane_counts)[len(frame_lane_counts) // 2] if frame_lane_counts else 0,
        "lane_length_unit": "image_pixels",
        "lane_lengths": image_lane_lengths,
        "relative_distance_unit": "image_pixels_only",
        "relative_distance": None,
        "relative_speed_unit": "pixels_per_second_only",
        "relative_speed": None,
        "note": "Absolute distance and vehicle speed require scale and ego-motion telemetry.",
    }
    validation = {
        "valid": False,
        "errors": ["Detector unavailable; no lane geometry was produced."] if detector_error else ["Metric calibration unavailable."],
        "warnings": [detector_error] if detector_error else [],
        "metrics": {"metric_status": metric_status, "open_drive_emitted": False},
    }
    payloads = {
        "input_metadata.json": {"path": meta.path, "width": meta.width, "height": meta.height, "fps": meta.fps,
                                 "frame_count": meta.frame_count, "duration": meta.duration,
                                 "processed_frames": processed, "processing_fps": processed / elapsed if elapsed else 0.0},
        "detection_results.json": {
            "observations": [
                {**obs.__dict__, "metric_validity": obs.metric_validity.value}
                for obs in all_observations
            ],
            "count": len(all_observations), "detector_error": detector_error,
        },
        "tracking_results.json": tracker.persist(),
        "lane_graph.json": graph.to_dict(),
        "geometry.json": {"status": metric_status, "segments": geometries},
        "metric_report.json": {"status": metric_status, "source": "monocular-video-only", "calibration_source": "missing",
                                "uncertainty": 1.0, "confidence": 0.0, "valid": False,
                                "relative_video_report": relative_report},
        "validation_report.json": validation,
        "run_summary.json": {"detector": detector.metadata(), "detector_error": detector_error,
                              "processed_frames": processed,
                              "persistent_track_count": len(tracks), "detected_observation_count": len(all_observations),
                              "detected_lane_count": len({track.lane_id for track in tracks}),
                              "topology_event_count": len(graph.events), "topology_events": graph.events,
                              "processing_fps": processed / elapsed if elapsed else 0.0,
                              "vehicle_speed_mps": None, "vehicle_speed_kmh": None, "vehicle_speed_status": "UNAVAILABLE_NO_TELEMETRY",
                              "distance_m": None, "road_length_m": None, "distance_status": "UNAVAILABLE_NO_SCALE",
                              "relative_video_report": relative_report,
                              "metric_status": metric_status, "open_drive_emitted": False},
    }
    for name, payload in payloads.items():
        write_json(str(root / name), payload)
    logging.info("Completed fail-closed run: %s", json.dumps(payloads["run_summary.json"]))
    return payloads["run_summary.json"]