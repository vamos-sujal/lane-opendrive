#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--repo-path", type=str, default="/content/Ultra-Fast-Lane-Detection")
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Smoke test requested CUDA, but CUDA is unavailable.")
    device = torch.device(args.device)
    tensor = torch.ones((2, 2), device=device)
    assert float(tensor.sum().item()) == 4.0
    detector_module = importlib.import_module("src.detection.ufld")
    assert detector_module.UFLDDetector.name == "ufld_culane"
    if not args.checkpoint:
        print(f"Smoke test passed on {device}: UFLD detector contract importable")
        return 0
    detector = detector_module.UFLDDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        repo_path=args.repo_path,
    )
    try:
        detector.load_weights()
        detector.warmup()
    except Exception as exc:
        print(f"UFLD checkpoint smoke test failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(2) from exc
    print(f"Smoke test passed on {device}: UFLD checkpoint loaded and warmed up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())