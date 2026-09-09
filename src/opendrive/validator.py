from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree


def validate_xodr(path: str) -> Dict[str, object]:
    errors: List[str] = []
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        return {"valid": False, "errors": [f"XML parse failed: {exc}"], "warnings": []}
    if root.tag != "OpenDRIVE":
        errors.append("Root element must be OpenDRIVE")
    roads = root.findall("road")
    if not roads:
        errors.append("No road elements")
    for road in roads:
        try:
            length = float(road.attrib["length"])
            if not math.isfinite(length) or length <= 0:
                errors.append(f"Invalid road length on road {road.attrib.get('id')}")
        except (KeyError, ValueError):
            errors.append("Road length is missing or non-numeric")
        if road.find("planView/geometry") is None:
            errors.append(f"Road {road.attrib.get('id')} has no planView geometry")
        if road.find("lanes/laneSection") is None:
            errors.append(f"Road {road.attrib.get('id')} has no laneSection")
    return {"valid": not errors, "errors": errors, "warnings": [], "road_count": len(roads)}