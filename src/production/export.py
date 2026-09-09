from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.opendrive.builder import build_xodr
from src.opendrive.validator import validate_xodr


def export_validated_opendrive(lane_boundaries: List[Dict[str, Any]], output: str, lane_width_m: float | None) -> Dict[str, Any]:
    geometries = []
    for boundary in lane_boundaries:
        points = boundary.get("points_world", [])
        if len(points) < 3:
            continue
        coefficients = __import__("numpy").polyfit(
            [float(point[1]) for point in points],
            [float(point[0]) for point in points],
            2,
        ).tolist()
        geometries.append({
            "lane_id": boundary["track_id"],
            "status": "METRIC_VALID",
            "points": points,
            "coefficients": coefficients,
        })
    try:
        build_xodr(geometries, output, lane_width_m)
        report = validate_xodr(output)
        if not report.get("valid"):
            Path(output).unlink(missing_ok=True)
        return report
    except (OSError, ValueError) as exc:
        Path(output).unlink(missing_ok=True)
        return {"valid": False, "errors": [str(exc)], "warnings": []}
