#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Smoke test requested CUDA, but CUDA is unavailable.")
    device = torch.device(args.device)
    tensor = torch.ones((2, 2), device=device)
    assert float(tensor.sum().item()) == 4.0
    detector = importlib.import_module("src.detection.ufld")
    assert detector.UFLDDetector.name == "ufld_culane"
    print(f"Smoke test passed on {device}: UFLD detector contract importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())