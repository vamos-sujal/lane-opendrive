#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
import gdown


def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify a reference lane detector checkpoint.")
    parser.add_argument("--output-dir", type=str, default="weights")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--expected-sha256", type=str, default="")
    parser.add_argument("--expected-md5", type=str, default="")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        checkpoint = config.get("checkpoint", {})
        args.url = args.url or checkpoint.get("url") or ""
        args.expected_sha256 = args.expected_sha256 or checkpoint.get("sha256") or ""
        args.expected_md5 = args.expected_md5 or checkpoint.get("md5") or ""
        configured_path = checkpoint.get("path")
    else:
        configured_path = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.url:
        raise SystemExit(
            "No official checkpoint URL supplied. Set checkpoint.url and checkpoint.sha256 "
            "in configs/model.yaml before running GPU inference."
        )

    destination = out_dir / (Path(configured_path).name if configured_path else "latr_checkpoint.pth")
    print(f"Download target: {args.url}")
    print(f"Output file: {destination}")
    if "drive.google.com" in args.url:
        downloaded = gdown.download(args.url, str(destination), quiet=False)
        if not downloaded:
            raise SystemExit("Google Drive checkpoint download failed.")
    else:
        import urllib.request
        urllib.request.urlretrieve(args.url, destination)

    if args.expected_sha256:
        actual = sha256sum(str(destination))
        if actual.lower() != args.expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise SystemExit(f"Checkpoint SHA256 mismatch: expected {args.expected_sha256}, got {actual}")
        print(f"Verified SHA256: {actual}")
    elif args.expected_md5:
        actual = md5sum(str(destination))
        if actual.lower() != args.expected_md5.lower():
            destination.unlink(missing_ok=True)
            raise SystemExit(f"Checkpoint MD5 mismatch: expected {args.expected_md5}, got {actual}")
        print(f"Verified MD5: {actual}")
    else:
        print("Warning: the official source does not publish a checksum for this checkpoint; source URL is recorded in manifest.json.")

    (out_dir / "manifest.json").write_text(
        json.dumps({
            "url": args.url,
            "path": str(destination),
            "sha256": args.expected_sha256 or None,
            "md5": args.expected_md5 or None,
        }, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
