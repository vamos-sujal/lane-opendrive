#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.video_reader import resolve_input_video
from src.production.model_registry import load_models
from src.production.runner import run_production


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the production multi-model road perception baseline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--scene-stride", type=int, default=5)
    parser.add_argument("--lane-checkpoint", type=str, default=None)
    parser.add_argument("--lane-repo-path", type=str, default="/content/Ultra-Fast-Lane-Detection")
    args = parser.parse_args()

    input_path = resolve_input_video(args.input)
    models = load_models(args.device)
    lane_detector = None
    if args.lane_checkpoint:
        from src.production.lane_detector import ProductionLaneDetector
        lane_detector = ProductionLaneDetector(args.lane_checkpoint, args.lane_repo_path)
        lane_detector.load()
    run_production(input_path, args.output, models, args.device, max(1, args.scene_stride), lane_detector)
    print(f"Production report written to {Path(args.output) / 'production_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
