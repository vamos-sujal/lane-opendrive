#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.production.model_registry import DEPTH_MODEL, OBJECT_MODEL, SCENE_MODEL, load_models, runtime_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the production T4 model stack before video inference.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    report = runtime_report()
    print(json.dumps(report, indent=2))
    if args.device == "cuda" and not report.get("cuda"):
        raise SystemExit("Production runtime check failed: CUDA is unavailable.")
    if any(str(value).startswith("ERROR:") for value in report["packages"].values()):
        raise SystemExit("Production runtime check failed: a required package could not be imported.")

    print(f"Loading object model: {OBJECT_MODEL}")
    print(f"Loading metric depth model: {DEPTH_MODEL}")
    print(f"Loading scene model: {SCENE_MODEL}")
    models = load_models(args.device)
    if models.object_model is None or models.depth_model is None or models.scene_model is None:
        raise SystemExit("Production runtime check failed: a model did not load.")
    print("Production model stack loaded successfully.")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
