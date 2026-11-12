"""Check exported STL topology and render the prototype grip kit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root = Path(__file__).parent
folder = root / "generated"
parameters = json.loads((root / "parameters.json").read_text())
manifest = json.loads((folder / "manifest.json").read_text())
meshes = {}
for name in manifest["parts"]:
    mesh = trimesh.load_mesh(folder / f"{name}.stl")
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split()) != 1:
        raise ValueError(f"invalid STL topology: {name}")
    meshes[name] = mesh

fig = plt.figure(figsize=(12, 7), facecolor="#f4f5f6")
ax = fig.add_subplot(121, projection="3d")
colors = ["#218e85", "#ef943f"]
gap = parameters["preview_closed_mount_separation"]
for name, offset, color in zip(
    ("right_thumb_grip", "right_finger_grip"), (-gap / 2, gap / 2), colors, strict=True
):
    mesh = meshes[name].copy()
    mesh.apply_translation([offset, 0, 0])
    collection = Poly3DCollection(
        mesh.triangles, facecolor=color, linewidth=0, edgecolor="none", alpha=1
    )
    ax.add_collection3d(collection)
ax.set(xlim=(-65, 65), ylim=(-20, 85), zlim=(0, 15))
ax.set_box_aspect((130, 105, 35))
ax.view_init(elev=55, azim=-80)
ax.set_axis_off()
ax.set_title("Right-hand pair · illustrative mounting", fontsize=12, pad=10)

ax2 = fig.add_subplot(122)
for name, offset, color in zip(
    ("right_thumb_grip", "right_finger_grip"), (-gap / 2, gap / 2), colors, strict=True
):
    mesh = meshes[name].copy()
    mesh.apply_translation([offset, 0, 0])
    # Top-facing triangles preserve the apertures and mounting slots.
    triangles = mesh.triangles[mesh.face_normals[:, 2] > 0.9]
    from matplotlib.collections import PolyCollection

    ax2.add_collection(
        PolyCollection(triangles[:, :, :2], facecolor=color, edgecolor=color, linewidth=0.15)
    )
ax2.set(xlim=(-65, 65), ylim=(-20, 85), xlabel="mm", ylabel="mm")
ax2.set_aspect("equal")
ax2.grid(alpha=0.15)
ax2.set_title("Top view · 8 mm thick", fontsize=12)
fig.suptitle("YAM scissor-style teaching grips", fontsize=22, fontweight="bold", y=0.95)
fig.text(
    0.5,
    0.865,
    "Two mirrored pairs + mounting coupon · STEP + STL · millimetres",
    ha="center",
    fontsize=11,
    color="#444444",
)
fig.text(
    0.5,
    0.08,
    "Thumb aperture 24 × 30 mm   |   Finger aperture 36 × 26 mm   |   Wall 5 mm",
    ha="center",
    fontsize=11,
)
fig.text(
    0.5,
    0.035,
    "PROTOTYPE — 14 mm mounting pitch and 38 mm assembly gap must be measured on your rig.",
    ha="center",
    fontsize=10,
    color="#8b4722",
)
fig.subplots_adjust(top=0.8, bottom=0.16, wspace=0.12)
fig.savefig(folder / "preview.png", dpi=180, facecolor=fig.get_facecolor())
print(json.dumps({"watertight_parts": len(meshes), "connected_parts": len(meshes)}, indent=2))
