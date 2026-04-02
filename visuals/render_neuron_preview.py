from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SCENE_PATH = ROOT / "neuron_scene_data.json"
OUTPUT_PATH = ROOT / "neuron_brain_3d.png"


def extract_scene_data() -> dict:
    if not SCENE_PATH.exists():
        raise RuntimeError("Scene JSON not found. Run generate_scene_assets.py first.")
    return json.loads(SCENE_PATH.read_text(encoding="utf-8"))


def render() -> Path:
    scene = extract_scene_data()
    nodes = scene["nodes"]
    edges = scene["edges"]
    groups = scene["groups"]
    node_map = {node["id"]: node for node in nodes}
    x_values = [node["x"] for node in nodes]
    y_values = [node["y"] for node in nodes]
    z_values = [node["z"] for node in nodes]
    x_pad = 70
    y_pad = 70
    z_pad = 90

    fig = plt.figure(figsize=(18, 13), facecolor="#07111a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#07111a")
    ax.view_init(elev=18, azim=-58)
    ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
    ax.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)
    ax.set_zlim(min(z_values) - z_pad, max(z_values) + z_pad)
    ax.set_axis_off()

    for edge in edges:
        start = node_map[edge[0]]
        end = node_map[edge[1]]
        ax.plot(
            [start["x"], end["x"]],
            [start["y"], end["y"]],
            [start["z"], end["z"]],
            color="#5ea7d8",
            alpha=0.18,
            linewidth=1.2,
        )

    for group_key, group in groups.items():
        group_nodes = [node for node in nodes if node["group"] == group_key]
        xs = [node["x"] for node in group_nodes]
        ys = [node["y"] for node in group_nodes]
        zs = [node["z"] for node in group_nodes]
        sizes = [node["size"] * 24 for node in group_nodes]
        ax.scatter(xs, ys, zs, s=[size * 2.2 for size in sizes], c=group["color"], alpha=0.07, depthshade=False)
        ax.scatter(xs, ys, zs, s=sizes, c=group["color"], alpha=0.94, depthshade=True, edgecolors="#f7fbff", linewidths=0.7)

    plt.tight_layout(pad=0)
    fig.savefig(OUTPUT_PATH, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = render()
    print(path)
