#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.common import MetricState
from src.io.video_reader import resolve_input_video
from src.pipeline import run_pipeline


def build_default_config() -> dict:
    return {
        "model": "latr",
        "checkpoint": None,
        "metric_status": MetricState.NON_METRIC.value,
        "safe_mode": True,
        "camera": {"calibration_available": False},
        "topology": {"enabled": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the lane-to-OpenDRIVE pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input MP4 file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/camera.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--detector", choices=("ufld_culane", "latr"), default="ufld_culane")
    args = parser.parse_args()

    resolved_input = resolve_input_video(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_dir / "visualizations", exist_ok=True)
    os.makedirs(out_dir / "logs", exist_ok=True)

    cfg = build_default_config()
    camera_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    cfg["camera"] = camera_config.get("camera", cfg["camera"])
    cfg["pipeline"] = camera_config.get("pipeline", {})
    cfg["input"] = resolved_input
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    try:
        summary = run_pipeline(resolved_input, str(out_dir), args.checkpoint, args.detector, cfg["camera"])
        print(f"Resolved input: {resolved_input}")
        print(f"Processed frames: {summary['processed_frames'] if 'processed_frames' in summary else 'see input_metadata.json'}")
        print(f"Persistent tracks: {summary['persistent_track_count']}")
        print(f"Detector error: {summary['detector_error'] or 'none'}")
        print(f"Metric status: {summary.get('metric_status', 'unknown')}")
        print(f"OpenDRIVE emitted: {summary.get('open_drive_emitted', False)}")
        return 0
    except Exception as exc:
        with open(out_dir / "logs" / "pipeline.log", "w", encoding="utf-8") as f:
            f.write(f"ERROR: {exc}\n")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
