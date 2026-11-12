"""Generate paired, direct-jaw scissor-style teaching grips; dimensions in mm.

The mounting pitch is deliberately a prototype parameter, not an asserted YAM
drawing dimension. Print the fit coupon and measure the actual jaw first.
Run with cadquery==2.8.0; no robot dependencies or network access are used.
"""

import argparse
import json
from pathlib import Path

import cadquery as cq


def validate(p):
    if p["units"] != "mm":
        raise ValueError("CAD parameters must use mm")
    for key, value in p.items():
        if key in {"units", "fit_status"}:
            continue
        if not isinstance(value, (float, int)) or not 0 < value < 1000:
            raise ValueError(f"invalid dimension {key}")
    if p["mount_slot_width"] >= p["mount_slot_length"]:
        raise ValueError("slot length must exceed width")
    if p["mount_slot_length"] + 6 > p["plate_width"]:
        raise ValueError("need at least 3 mm material outside mount slots")
    if p["mount_pitch"] + p["mount_slot_width"] + 6 > p["plate_length"]:
        raise ValueError("mount slots too close to plate edge")
    if p["ring_wall"] < 4 or p["thickness"] < 6:
        raise ValueError("prototype minimum wall 4 mm, thickness 6 mm")


def plate(p):
    shape = cq.Workplane("XY").rect(p["plate_width"], p["plate_length"]).extrude(p["thickness"])
    shape = shape.edges("|Z").fillet(3)
    for y in (-p["mount_pitch"] / 2, p["mount_pitch"] / 2):
        slot = (
            cq.Workplane("XY")
            .center(0, y)
            .slot2D(p["mount_slot_length"], p["mount_slot_width"])
            .extrude(p["thickness"])
        )
        shape = shape.cut(slot)
    return shape


def grip(p, *, thumb):
    width = p["thumb_inner_width"] if thumb else p["finger_inner_width"]
    length = p["thumb_inner_length"] if thumb else p["finger_inner_length"]
    x, y, wall, thick = p["ring_offset_x"], p["ring_offset_y"], p["ring_wall"], p["thickness"]
    # Wide tapered neck, overlapping the plate and the outer ring; union then
    # cut the aperture so the neck cannot obstruct the finger opening.
    neck = (
        cq.Workplane("XY")
        .polyline(
            [
                (-p["neck_width"] / 2, 10),
                (p["neck_width"] / 2, 10),
                (x + p["neck_width"] / 2, y - length / 2 - 1),
                (x - p["neck_width"] / 2, y - length / 2 - 1),
            ]
        )
        .close()
        .extrude(thick)
    )
    outer = (
        cq.Workplane("XY").center(x, y).ellipse(width / 2 + wall, length / 2 + wall).extrude(thick)
    )
    hole = cq.Workplane("XY").center(x, y).ellipse(width / 2, length / 2).extrude(thick)
    # Keep ring edges planar for robust exports; ease skin-contact edges after printing.
    ring = outer.cut(hole)
    shape = plate(p).union(neck).union(ring).cut(hole)
    return shape


def build(parameters, output):
    p = json.loads(Path(parameters).read_text())
    validate(p)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thumb, fingers = grip(p, thumb=True).mirror("YZ"), grip(p, thumb=False)
    parts = {
        "right_thumb_grip": thumb,
        "right_finger_grip": fingers,
        "left_thumb_grip": thumb.mirror("YZ"),
        "left_finger_grip": fingers.mirror("YZ"),
        "fit_coupon": plate(p),
    }
    manifest = {"units": "mm", "fit_status": p["fit_status"], "parts": {}}
    for name, part in parts.items():
        solid = part.val()
        if not solid.isValid() or len(solid.Solids()) != 1 or solid.Volume() <= 0:
            raise ValueError(f"invalid or disconnected CAD solid: {name}")
        cq.exporters.export(part, str(output / f"{name}.step"))
        cq.exporters.export(part, str(output / f"{name}.stl"), tolerance=0.08, angularTolerance=0.1)
        bounds = solid.BoundingBox()
        manifest["parts"][name] = {
            "solids": 1,
            "volume_mm3": solid.Volume(),
            "bounds_mm": [bounds.xlen, bounds.ylen, bounds.zlen],
            "estimated_petg_g_at_1_27_g_cm3_solid": solid.Volume() * 0.00127,
        }
    assembly = cq.Assembly(name="YAM_scissor_style_grips_prototype")
    gap = p["preview_closed_mount_separation"]
    assembly.add(
        thumb,
        name="thumb",
        loc=cq.Location(cq.Vector(-gap / 2, 0, 0)),
        color=cq.Color(0.2, 0.65, 0.6),
    )
    assembly.add(
        fingers,
        name="fingers",
        loc=cq.Location(cq.Vector(gap / 2, 0, 0)),
        color=cq.Color(0.98, 0.55, 0.2),
    )
    assembly.export(str(output / "assembly.step"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters", type=Path, default=Path(__file__).with_name("parameters.json")
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("generated"))
    args = parser.parse_args()
    print(json.dumps(build(args.parameters, args.output), indent=2))
