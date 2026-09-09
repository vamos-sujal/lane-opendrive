from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List
from xml.etree.ElementTree import Element, ElementTree, SubElement

from src.common import MetricState


def build_xodr(geometries: List[Dict[str, object]], output: str, lane_width_m: float | None = None) -> None:
    if not geometries or any(item.get("status") != MetricState.METRIC_VALID.value for item in geometries):
        raise ValueError("Refusing OpenDRIVE generation: metric geometry is unavailable or invalid.")
    if lane_width_m is None or lane_width_m <= 0:
        raise ValueError("Refusing OpenDRIVE generation: a measured lane_width_m is required.")
    if len(geometries) < 2:
        raise ValueError("Refusing OpenDRIVE generation: at least two lane boundaries are required.")
    points = [point for item in geometries for point in item.get("points", [])]
    if not points:
        raise ValueError("Refusing OpenDRIVE generation: no metric lane points are available.")
    y_start = max(min(float(point[1]) for point in item["points"]) for item in geometries)
    y_end = min(max(float(point[1]) for point in item["points"]) for item in geometries)
    if y_end <= y_start:
        raise ValueError("Refusing OpenDRIVE generation: lane boundaries have no common metric span.")

    def evaluate(item: Dict[str, object], y_value: float) -> float:
        coefficients = [float(value) for value in item.get("coefficients", [])]
        if not coefficients:
            raise ValueError("Refusing OpenDRIVE generation: a lane boundary has no fitted polynomial.")
        result = 0.0
        for coefficient in coefficients:
            result = result * y_value + coefficient
        return result

    samples = [y_start + (y_end - y_start) * index / 31.0 for index in range(32)]
    centerline = []
    for y_value in samples:
        boundaries = sorted(evaluate(item, y_value) for item in geometries)
        spacings = [right - left for left, right in zip(boundaries, boundaries[1:])]
        if any(spacing <= 0.0 or abs(spacing - lane_width_m) > max(0.75, lane_width_m * 0.5) for spacing in spacings):
            raise ValueError("Refusing OpenDRIVE generation: lane spacing conflicts with lane_width_m.")
        centerline.append(((boundaries[0] + boundaries[-1]) / 2.0, y_value))
    road_length = sum(math.hypot(x2 - x1, y2 - y1)
                      for (x1, y1), (x2, y2) in zip(centerline, centerline[1:]))
    if road_length <= 0:
        raise ValueError("Refusing OpenDRIVE generation: metric road length is not positive.")
    root = Element("OpenDRIVE")
    header = SubElement(root, "header", revMajor="1", revMinor="6", name="lane-opendrive", version="1.00")
    road = SubElement(root, "road", name="lane-road", length=f"{road_length:.6f}", id="1", junction="-1")
    plan_view = SubElement(road, "planView")
    distance = 0.0
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(centerline, centerline[1:])):
        segment_length = math.hypot(x2 - x1, y2 - y1)
        geometry = SubElement(plan_view, "geometry", s=f"{distance:.6f}", x=f"{x1:.6f}", y=f"{y1:.6f}",
                              hdg=f"{math.atan2(y2 - y1, x2 - x1):.9f}", length=f"{segment_length:.6f}")
        geometry.append(Element("line"))
        distance += segment_length
    lanes = SubElement(road, "lanes")
    section = SubElement(lanes, "laneSection", s="0.0")
    center = SubElement(section, "center")
    SubElement(center, "lane", id="0", type="none", level="false")
    left = SubElement(section, "left")
    for index in range(1, len(geometries)):
        lane = SubElement(left, "lane", id=str(index), type="driving", level="false")
        SubElement(lane, "width", sOffset="0.0", a=f"{lane_width_m:.6f}", b="0.0", c="0.0", d="0.0")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)