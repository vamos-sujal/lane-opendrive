from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from xml.etree.ElementTree import Element, ElementTree, SubElement

from src.common import MetricState


def build_xodr(geometries: List[Dict[str, object]], output: str) -> None:
    if not geometries or any(item.get("status") != MetricState.METRIC_VALID.value for item in geometries):
        raise ValueError("Refusing OpenDRIVE generation: metric geometry is unavailable or invalid.")
    root = Element("OpenDRIVE")
    header = SubElement(root, "header", revMajor="1", revMinor="6", name="lane-opendrive", version="1.00")
    road = SubElement(root, "road", name="lane-road", length="1.0", id="1", junction="-1")
    plan_view = SubElement(road, "planView")
    SubElement(plan_view, "geometry", s="0.0", x="0.0", y="0.0", hdg="0.0", length="1.0").append(Element("line"))
    lanes = SubElement(road, "lanes")
    section = SubElement(lanes, "laneSection", s="0.0")
    center = SubElement(section, "center")
    SubElement(center, "lane", id="0", type="none", level="false")
    left = SubElement(section, "left")
    for index, _ in enumerate(geometries, 1):
        lane = SubElement(left, "lane", id=str(index), type="driving", level="false")
        SubElement(lane, "width", sOffset="0.0", a="1.0", b="0.0", c="0.0", d="0.0")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)