#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys


REQUIRED = ("cv2", "numpy", "scipy", "yaml", "lxml", "shapely", "torch")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    for module_name in REQUIRED:
        module = importlib.import_module(module_name)
        print(f"{module_name}: {getattr(module, '__version__', 'imported')}")
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Colab/T4 execution target.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())