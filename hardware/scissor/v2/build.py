"""Flash-style long/short leads on removable YAM jaw-root split clamps.

Assembly frame (mm): X jaw travel, Y toward fingertips, Z up.
No factory mounting-hole dimensions are assumed. Clamp dimensions are a
prototype envelope of the pinned i2rt linear_4310 visual mesh, not a drawing.
"""

import argparse
import json
import math
from pathlib import Path

import cadquery as cq
import trimesh

HERE = Path(__file__).resolve().parent


def block(x0, x1, y0, y1, z0, z1):
    return (
        cq.Workplane("XY")
        .box(x1 - x0, y1 - y0, z1 - z0)
        .translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    )


def yz_plane(x):
    return cq.Plane(origin=(x, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))


def yz_poly(points, x, thickness):
    return cq.Workplane(yz_plane(x)).polyline(points).close().extrude(thickness)


def ease(part, radius=0.8):
    # Apply before booleans to retain rounded skin-contact edges without
    # masking CAD failures with broad exception handling.
    return part.edges().fillet(radius)


def clamp(p):
    x0, x1 = p["clamp_outer_x"], p["clamp_inner_wall_x"]
    y0, y1 = p["jaw_root_y_start"], p["jaw_root_y_end"]
    z0 = p["clamp_bottom_z"]
    wall = block(x0, x1, y0, y1, z0, p["wall_top_z"])
    lip = block(x0, p["clamp_lip_x"], y0, y1, z0, p["jaw_cavity_bottom_z"])
    base = wall.union(lip).edges("|Z").fillet(0.65)
    cap = ease(block(x0, p["clamp_lip_x"], y0, y1, p["jaw_cavity_top_z"], p["cap_top_z"]), 0.65)
    for y in p["bolt_y"]:
        hole = (
            cq.Workplane("XY")
            .center(p["bolt_x"], y)
            .circle(p["bolt_clearance"] / 2)
            .extrude(30)
            .translate((0, 0, 16))
        )
        base = base.cut(hole)
        cap = cap.cut(hole)
        # Side-loading M3 hex nut pocket. Slot opens on the outer X face.
        nut = (
            cq.Workplane("XY")
            .center(p["bolt_x"], y)
            .polygon(6, p["nut_across_flats"] / math.cos(math.pi / 6))
            .extrude(2.8)
            .translate((0, 0, 22))
        )
        slot = block(x0 - 1, p["bolt_x"], y - 2.9, y + 2.9, 22, 24.8)
        base = base.cut(nut.union(slot))
        counterbore = (
            cq.Workplane("XY").center(p["bolt_x"], y).circle(3.1).extrude(4).translate((0, 0, 38.6))
        )
        cap = cap.cut(counterbore)
    return base, cap


def pitch(part, p):
    return part.rotate((0, 79, -40), (1, 79, -40), -p["handle_backward_pitch_deg"])


def lead(p, thumb):
    x, t = p["handle_x_start"], p["handle_thickness"]
    neck = ease(block(x, x + t, 72, 86, -62, -37), 1.0)
    if thumb:
        outer = (
            cq.Workplane(yz_plane(x))
            .center(69, -76)
            .circle(p["thumb_inner_radius"] + p["thumb_wall"])
            .extrude(t)
        )
        inner = (
            cq.Workplane(yz_plane(x - 1))
            .center(69, -76)
            .circle(p["thumb_inner_radius"])
            .extrude(t + 2)
        )
        shell = outer.cut(inner).cut(block(x - 1, x + t + 1, 20, 69, -87, -65))
        shell = ease(shell, 0.9)
        part = neck.union(shell)
        slots = [(69, -57.5), (69, -94.5)]
        for y, z in slots:
            part = part.union(ease(block(x, x + t, y - 9, y + 9, z - 4, z + 4), 1))
    else:
        # Open-backed paddle: long spine, rounded upper/lower hooks, and
        # four shallow finger scallops, following the reference lead layout.
        spine = block(x, x + t, 69, 86, -151, -51)
        top = block(x, x + t, 50, 86, -63, -51)
        bottom = block(x, x + t, 50, 86, -154, -142)
        part = spine.union(top).union(bottom).edges("|X").fillet(4)
        for z in (-73, -94, -115, -136):
            scallop = cq.Workplane(yz_plane(x - 1)).center(64, z).circle(7).extrude(t + 2)
            part = part.cut(scallop)
        part = part.edges("not |X").fillet(0.75).union(neck)
        slots = [(67, -57), (77, -98), (77, -107), (67, -148)]
    for y, z in slots:
        slot = (
            cq.Workplane(yz_plane(x - 1))
            .center(y, z)
            .slot2D(p["strap_slot_width"], p["strap_slot_height"])
            .extrude(t + 2)
        )
        part = part.cut(slot)
    return pitch(part, p)


def strap(p, y0, z0, y1, z1, bulge):
    # Schematic flexible ribbon; excluded from printable parts and collision
    # claims. Endpoints lie in the lead's strap slots; adjust actual Velcro.
    points = []
    for i in range(41):
        u = i / 40
        points.append((y0 + (y1 - y0) * u - bulge * math.sin(math.pi * u), z0 + (z1 - z0) * u))
    contour = [(y - p["strap_thickness_visual"] / 2, z) for y, z in points]
    contour += [(y + p["strap_thickness_visual"] / 2, z) for y, z in reversed(points)]
    x = p["handle_x_start"] + (p["handle_thickness"] - p["strap_width"]) / 2
    return pitch(yz_poly(contour, x, p["strap_width"]), p)


def build(p, out):
    out.mkdir(parents=True, exist_ok=True)
    base, cap = clamp(p)
    # Both leads are modeled on the negative-X jaw; mirror for the other jaw.
    thumb, fingers = base.union(lead(p, True)), base.union(lead(p, False))
    parts = {
        "right_thumb_lead": thumb,
        "right_finger_lead": fingers.mirror("YZ"),
        "left_thumb_lead": thumb.mirror("YZ"),
        "left_finger_lead": fingers,
        "negative_jaw_cap": cap,
        "positive_jaw_cap": cap.mirror("YZ"),
        "fit_coupon_base": base,
    }
    assembly = cq.Assembly(name="YAM_Flash_leads_v2_right_hand")
    colors = {"right_thumb_lead": (0.9, 0.17, 0.27), "right_finger_lead": (0.06, 0.48, 0.92)}
    report = {"units": "mm", "status": p["cad_fit_status"], "parts": {}}
    for name, part in parts.items():
        solid = part.val()
        if not solid.isValid() or len(solid.Solids()) != 1 or solid.Volume() <= 0:
            raise ValueError(f"Invalid or disconnected part: {name}")
        cq.exporters.export(part, str(out / f"{name}.step"))
        cq.exporters.export(part, str(out / f"{name}.stl"), tolerance=0.06, angularTolerance=0.1)
        # Binary STL collapses a few fillet-pole triangles to zero area.
        # Remove only duplicate/degenerate faces, without filling or remeshing.
        mesh = trimesh.load_mesh(out / f"{name}.stl")
        mesh.merge_vertices(digits_vertex=6)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split()) != 1:
            raise ValueError(f"Invalid STL export: {name}")
        mesh.export(out / f"{name}.stl")
        bb = solid.BoundingBox()
        report["parts"][name] = {
            "cad_valid": True,
            "solid_count": 1,
            "volume_mm3": solid.Volume(),
            "solid_petg_g": solid.Volume() * 0.00127,
            "bounds_mm": [[bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymax, bb.zmax]],
        }
        if name.startswith("right_") or name.endswith("jaw_cap"):
            assembly.add(part, name=name, color=cq.Color(*colors.get(name, (0.24, 0.29, 0.35))))
    references = {
        "thumb_strap": strap(p, 69, -57.5, 69, -94.5, 27),
        "finger_strap_upper": strap(p, 67, -57, 77, -98, 33).mirror("YZ"),
        "finger_strap_lower": strap(p, 77, -107, 67, -148, 33).mirror("YZ"),
    }
    for name, part in references.items():
        cq.exporters.export(part, str(out / f"reference_{name}.stl"), tolerance=0.08)
        assembly.add(part, name=name, color=cq.Color(0.10, 0.13, 0.17))
    assembly.export(str(out / "right_hand_assembly.step"))
    left = cq.Assembly(name="YAM_Flash_leads_v2_left_hand")
    for name in ("left_thumb_lead", "left_finger_lead", "negative_jaw_cap", "positive_jaw_cap"):
        left.add(parts[name], name=name)
    left.export(str(out / "left_hand_assembly.step"))
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exported {len(parts)} valid solids and two STEP assemblies to {out}")
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=HERE / "parameters.json")
    parser.add_argument("--output", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    build(json.loads(args.parameters.read_text()), args.output)
