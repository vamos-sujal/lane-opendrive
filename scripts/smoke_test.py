#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib


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
    detector.load_weights()
    detector.warmup()
    print(f"Smoke test passed on {device}: UFLD checkpoint loaded and warmed up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())