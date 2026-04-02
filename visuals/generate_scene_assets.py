from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_brain.neurons import ALL_NEURONS


SCENE_JS = ROOT / "neuron_scene_data.js"
SCENE_JSON = ROOT / "neuron_scene_data.json"


GROUPS = {
    "core": {"label": "Core Identity", "color": "#66d9ef"},
    "market": {"label": "Market Read", "color": "#4dd4a0"},
    "signal": {"label": "Signal Matrix", "color": "#7de2d1"},
    "quality": {"label": "Quality Filters", "color": "#8db9ff"},
    "adaptive": {"label": "Adaptive Cortex", "color": "#bca7ff"},
    "cortex": {"label": "Meta Cortex", "color": "#d0c6ff"},
    "pro": {"label": "Pro Desk Read", "color": "#9aa8ff"},
    "setup": {"label": "Setup Forge", "color": "#ffbe76"},
    "plan_micro": {"label": "Plan Lab", "color": "#ffd6a5"},
    "risk_micro": {"label": "Risk Micro", "color": "#f4978e"},
    "risk": {"label": "Risk Desk", "color": "#ff7b72"},
    "output": {"label": "Output Gate", "color": "#f4f1de"},
}

STAGE_ORDER = [
    "core",
    "market",
    "signal",
    "quality",
    "adaptive",
    "cortex",
    "pro",
    "setup",
    "plan_micro",
    "risk_micro",
    "risk",
    "output",
]


def _ring_slots(count: int) -> list[tuple[float, float]]:
    slots: list[tuple[float, float]] = []
    if count == 0:
        return slots

    capacities = [1, 8, 14, 20, 28, 36]
    placed = 0
    ring_index = 0
    angle_seed = 0.0
    while placed < count:
        capacity = capacities[min(ring_index, len(capacities) - 1)]
        ring_count = min(capacity, count - placed)
        radius = 22 + ring_index * 44
        for i in range(ring_count):
            angle = angle_seed + (i / max(1, ring_count)) * 6.28318530718
            x = radius * math.cos(angle)
            y = radius * 0.74 * math.sin(angle)
            slots.append((x, y))
        placed += ring_count
        ring_index += 1
        angle_seed += 0.22
    return slots


def build_scene() -> dict:
    stage_buckets: dict[str, list] = {stage: [] for stage in STAGE_ORDER}
    fallback_stage = "risk"

    for neuron in ALL_NEURONS:
        stage = getattr(neuron, "visual_stage", fallback_stage)
        if stage not in stage_buckets:
            stage = fallback_stage
        stage_buckets[stage].append(neuron)

    stage_positions = {
        stage: -520 + idx * 120
        for idx, stage in enumerate(STAGE_ORDER)
    }

    nodes: list[dict] = []
    stage_node_ids: dict[str, list[str]] = {stage: [] for stage in STAGE_ORDER}

    index = 1
    for stage in STAGE_ORDER:
        bucket = stage_buckets[stage]
        slots = _ring_slots(len(bucket))
        for item, (x, y) in zip(bucket, slots):
            node_id = f"n{index:03d}"
            group = getattr(item, "visual_group", "risk")
            title = getattr(item, "title", item.name.replace("_", " ").title())
            description = getattr(item, "description", item.name)
            size = 12
            if group in {"core", "output"}:
                size = 16
            elif group in {"risk", "setup", "pro", "adaptive", "cortex"}:
                size = 14
            if index == len(ALL_NEURONS):
                size = 20
            node = {
                "id": node_id,
                "index": f"{index:03d}",
                "title": title,
                "group": group,
                "stage": stage,
                "x": round(x, 2),
                "y": round(y, 2),
                "z": stage_positions[stage],
                "size": size,
                "description": description,
            }
            nodes.append(node)
            stage_node_ids[stage].append(node_id)
            index += 1

    edges: list[list[str]] = []
    for stage in STAGE_ORDER:
        ids = stage_node_ids[stage]
        for left, right in zip(ids, ids[1:]):
            edges.append([left, right])

    for current_stage, next_stage in zip(STAGE_ORDER, STAGE_ORDER[1:]):
        current_ids = stage_node_ids[current_stage]
        next_ids = stage_node_ids[next_stage]
        if not current_ids or not next_ids:
            continue
        for idx, node_id in enumerate(current_ids):
            mapped = int(idx * len(next_ids) / max(1, len(current_ids)))
            mapped = min(mapped, len(next_ids) - 1)
            edges.append([node_id, next_ids[mapped]])
            if mapped + 1 < len(next_ids):
                edges.append([node_id, next_ids[mapped + 1]])

    scene = {
        "nodes": nodes,
        "edges": edges,
        "groups": GROUPS,
        "stage_order": STAGE_ORDER,
    }
    return scene


def write_scene(scene: dict) -> None:
    payload = json.dumps(scene, ensure_ascii=False)
    SCENE_JSON.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    SCENE_JS.write_text(f"window.NEURON_SCENE_DATA = {payload};\n", encoding="utf-8")


if __name__ == "__main__":
    scene = build_scene()
    write_scene(scene)
    print(SCENE_JSON)
