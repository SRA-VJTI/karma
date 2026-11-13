"""Load pinned i2rt references, verify rigid clearances, render real mesh geometry.

Requires the CAD requirements plus scipy, numba, pillow, vtk. No CAD/image AI.
Usage: python inspect_assembly.py --i2rt /path/to/i2rt
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "parameters.json").read_text())
TRAVEL = P["jaw_travel_each"]
POSITIONS = np.linspace(0, TRAVEL, math.ceil(TRAVEL / 0.5) + 1)
R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
COLORS = {
    "gripper": "#303c4b",
    "tip_left": "#d6dce1",
    "tip_right": "#d6dce1",
    "camera": "#697d91",
    "right_thumb_lead": "#e94060",
    "right_finger_lead": "#178ee9",
    "negative_jaw_cap": "#ad2d49",
    "positive_jaw_cap": "#1766aa",
}


def tf(e):
    t = np.eye(4)
    t[:3, 3] = np.fromstring(e.get("pos", "0 0 0"), sep=" ") * 1000
    q = np.fromstring(e.get("quat", "1 0 0 0"), sep=" ")
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return t


def references(root):
    source = root / "i2rt/robot_models/gripper/linear_4310"
    body = ET.parse(source / "linear_4310.xml").getroot().find("worldbody/body")
    out = {}
    for b in [body, *body.findall("body")]:
        g = b.find("geom")
        m = trimesh.load_mesh(source / "assets" / f"{g.get('mesh')}.stl")
        m.apply_scale(1000)
        m.apply_transform(tf(g) if b is body else tf(b) @ tf(g))
        m.vertices = m.vertices @ R.T
        out[b.get("name")] = m
    # Camera frame from the station model. Convert station gripper coordinates
    # back to standalone linear_4310 coordinates by registering the shared
    # gripper geom, rather than guessing the camera's pose.
    station = root / "i2rt/robot_models/station"
    sb = (
        ET.parse(station / "yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml")
        .getroot()
        .find(".//body[@name='left_gripper']")
    )
    cam = sb.find("body[@name='left_camera']")
    m = trimesh.load_mesh(station / "assets/d405_wrist_linear_4310.stl")
    m.apply_scale(1000)
    m.apply_transform(tf(body.find("geom")) @ np.linalg.inv(tf(sb.find("geom"))) @ tf(cam))
    m.vertices = m.vertices @ R.T
    out["camera"] = m
    return out


def load_parts():
    return {p.stem: trimesh.load_mesh(p) for p in (HERE / "generated").glob("*.stl")}


def scene(ref, parts, q=0, exploded=False, left=False):
    result = []
    for name, mesh in ref.items():
        mesh = mesh.copy()
        if name == "tip_left":
            mesh.apply_translation([-q, 0, 0])
        if name == "tip_right":
            mesh.apply_translation([q, 0, 0])
        result.append((name, mesh, COLORS[name]))
    names = [
        "right_thumb_lead",
        "right_finger_lead",
        "negative_jaw_cap",
        "positive_jaw_cap",
        "reference_thumb_strap",
        "reference_finger_strap_upper",
        "reference_finger_strap_lower",
    ]
    for name in names:
        m = parts[name].copy()
        sign = (
            -1 if name in {"right_thumb_lead", "negative_jaw_cap", "reference_thumb_strap"} else 1
        )
        m.apply_translation([sign * q, 0, 0])
        if exploded:
            if name.endswith("jaw_cap"):
                m.apply_translation([sign * 12, 0, 35])
            else:
                m.apply_translation([sign * 18, 0, -12])
        if left:
            m.apply_transform(np.diag([-1, 1, 1, 1]))
        result.append((name, m, COLORS.get(name, "#242d3b")))
    # Unthreaded hardware envelopes show seating and wrench access, not threads.
    for sign in (-1, 1):
        for y in P["bolt_y"]:
            x = sign * (abs(P["bolt_x"]) + q)
            z_extra = 35 if exploded else 0
            x_extra = sign * 12 if exploded else 0
            shaft = trimesh.creation.cylinder(radius=1.5, height=20, sections=20)
            shaft.apply_translation([x + x_extra, y, 28.6 + z_extra])
            head = trimesh.creation.cylinder(radius=2.75, height=3, sections=32)
            head.apply_translation([x + x_extra, y, 40.1 + z_extra])
            result += [("M3_shaft", shaft, "#8c99a8"), ("M3_head", head, "#a6b2bf")]
    return result


@njit(cache=True)
def raster(vertices, triangles, shades, width, height):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = 240
    pixels[:, :, 1] = 244
    pixels[:, :, 2] = 249
    depth = np.full((height, width), -1.0e20)
    for i in range(len(triangles)):
        a, b, c = vertices[triangles[i, 0]], vertices[triangles[i, 1]], vertices[triangles[i, 2]]
        xmin = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        xmax = min(width - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        ymin = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        ymax = min(height - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1.0e-9:
            continue
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                w0 = ((b[1] - c[1]) * (x + 0.5 - c[0]) + (c[0] - b[0]) * (y + 0.5 - c[1])) / den
                w1 = ((c[1] - a[1]) * (x + 0.5 - c[0]) + (a[0] - c[0]) * (y + 0.5 - c[1])) / den
                w2 = 1 - w0 - w1
                if w0 >= -1.0e-6 and w1 >= -1.0e-6 and w2 >= -1.0e-6:
                    z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                    if z > depth[y, x]:
                        depth[y, x] = z
                        for k in range(3):
                            pixels[y, x, k] = shades[i, k]
    return pixels


def render(items, direction, size=(1200, 1100), padding=1.16):
    eye = np.array(direction, float)
    eye /= np.linalg.norm(eye)
    right = np.cross([0, 0, 1], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    rotation = np.array([right, -up, eye])
    vertices, faces, shades = [], [], []
    light = np.array([-0.5, 0.7, 1.3])
    light /= np.linalg.norm(light)
    offset = 0
    for _name, mesh, color in items:
        rgb = np.array([int(color[j : j + 2], 16) for j in (1, 3, 5)])
        n = mesh.face_normals
        diffuse = np.clip(n @ light, 0, 1)
        fill = np.clip(n @ eye, 0, 1)
        sh = np.clip(
            rgb[None, :] * (0.50 + 0.38 * diffuse[:, None] + 0.15 * fill[:, None]), 0, 255
        ).astype(np.uint8)
        vertices.append(mesh.vertices @ rotation.T)
        faces.append(mesh.faces + offset)
        shades.append(sh)
        offset += len(mesh.vertices)
    v = np.vstack(vertices)
    f = np.vstack(faces)
    s = np.vstack(shades)
    bounds = np.array([v.min(0), v.max(0)])
    center = bounds.mean(0)
    scale = (
        min(size[0] / (bounds[1, 0] - bounds[0, 0]), size[1] / (bounds[1, 1] - bounds[0, 1]))
        / padding
    )
    v[:, :2] = (v[:, :2] - center[:2]) * scale + np.array(size) / 2
    return Image.fromarray(raster(v, f, s, *size))


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for f in candidates:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def board(ref, parts, output):
    canvas = Image.new("RGB", (2400, 2540), "#ffffff")
    d = ImageDraw.Draw(canvas)
    d.text((70, 35), "YAM / FLASH-STYLE TEACHING GRIPS", font=font(49, True), fill="#17273b")
    d.text(
        (70, 102),
        "V2 · Split jaw clamps · Short thumb saddle + long finger lead",
        font=font(29),
        fill="#52667d",
    )
    views = [
        (0, (-1.3, 2.4, 1.0), False, "01  CLOSED", "48 mm nominal outer grip span"),
        (
            TRAVEL,
            (-1.3, 2.4, 1.0),
            False,
            "02  FULL MECHANICAL TRAVEL",
            f"{2 * TRAVEL:g} mm added opening · not a hand-span claim",
        ),
        (
            10,
            (1, 0.03, 0.06),
            False,
            "03  SIDE / HAND INTERFACE",
            f"{P['handle_backward_pitch_deg']:g}° backward pitch · three adjustable strap loops",
        ),
        (
            12,
            (-1.5, 2.0, 1.1),
            True,
            "04  EXPLODED MOUNT",
            "Removable caps · captive M3 nuts · no jaw drilling",
        ),
    ]
    for idx, (q, direction, explode, title, sub) in enumerate(views):
        x = 40 + (idx % 2) * 1200
        y = 175 + (idx // 2) * 1120
        canvas.paste(render(scene(ref, parts, q, explode), direction, (1120, 975)), (x, y + 76))
        d.text((x + 20, y), title, font=font(30, True), fill="#23394e")
        d.text((x + 20, y + 40), sub, font=font(24), fill="#5d7188")
    d.text(
        (70, 2440),
        "Actual i2rt linear_4310 + wrist camera meshes. All colored rigid parts are exported CAD.",
        font=font(26),
        fill="#334960",
    )
    d.text(
        (70, 2482),
        "Prototype: physical clamp fit, hand sizing and load capacity still require bench checks.",
        font=font(25),
        fill="#985229",
    )
    canvas.save(output)
    hero = render(scene(ref, parts, 8), (-1.4, 2.5, 1.0), (2000, 1800), 1.15)
    hero.save(output.with_name("hero.png"))


def vtk_poly(mesh):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    p = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    p.SetPoints(points)
    cells = vtk.vtkCellArray()
    packed = np.column_stack([np.full(len(mesh.faces), 3), mesh.faces]).astype(np.int64).ravel()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(packed, deep=True))
    p.SetPolys(cells)
    return p


def swept_candidates(a, b, displacement):
    """Conservative broad phase in B's frame; never removes a possible contact.

    Each A triangle's AABB encloses its whole linear X trajectory. Alternate
    pruning against the surviving other mesh's bounds to remove unrelated
    regions (e.g. below-body handles and above-handle camera brackets).
    """
    al, ah = a.triangles.min(axis=1), a.triangles.max(axis=1)
    bl, bh = b.triangles.min(axis=1), b.triangles.max(axis=1)
    al[:, 0] += min(0, displacement)
    ah[:, 0] += max(0, displacement)
    ai, bi = np.arange(len(al)), np.arange(len(bl))
    for _ in range(12):
        old = (len(ai), len(bi))
        if not len(ai) or not len(bi):
            break
        ai = ai[
            np.all(ah[ai] >= bl[bi].min(0) - 1e-6, axis=1)
            & np.all(al[ai] <= bh[bi].max(0) + 1e-6, axis=1)
        ]
        if not len(ai):
            break
        bi = bi[
            np.all(bh[bi] >= al[ai].min(0) - 1e-6, axis=1)
            & np.all(bl[bi] <= ah[ai].max(0) + 1e-6, axis=1)
        ]
        if old == (len(ai), len(bi)):
            break
    return (
        trimesh.Trimesh(vertices=a.vertices, faces=a.faces[ai], process=False),
        trimesh.Trimesh(vertices=b.vertices, faces=b.faces[bi], process=False),
    )


def checker_controls():
    """Small geometric controls independently exercise pruning and VTK."""
    import vtk

    box = trimesh.creation.box()
    overlapping = box.copy()
    overlapping.apply_translation([0.75, 0, 0])
    ma, mb = swept_candidates(box, overlapping, 0)
    assert len(ma.faces) and len(mb.faces)
    check = vtk.vtkCollisionDetectionFilter()
    check.SetInputData(0, vtk_poly(ma))
    check.SetInputData(1, vtk_poly(mb))
    check.SetTransform(0, vtk.vtkTransform())
    check.SetTransform(1, vtk.vtkTransform())
    check.SetCollisionModeToFirstContact()
    check.Update()
    assert check.GetNumberOfContacts() > 0, "Intersection control failed"
    moving_target = box.copy()
    moving_target.apply_translation([3, 0, 0])
    ma, mb = swept_candidates(box, moving_target, 3)
    assert len(ma.faces) and len(mb.faces), "Crossing trajectory was incorrectly pruned"
    separated = box.copy()
    separated.apply_translation([0, 4, 0])
    ma, mb = swept_candidates(box, separated, 10)
    assert not len(ma.faces) or not len(mb.faces), "Separated swept bounds control failed"
    return {
        "known_intersection_detected": True,
        "crossing_motion_retained": True,
        "separated_motion_pruned": True,
    }


def verify(ref, parts, output):
    import vtk

    report = {
        "reference": "i2rt linear_4310 + station D405 camera",
        "units": "mm",
        "method": (
            "Conservative swept triangle-AABB pruning, then VTK surface "
            "intersection at 0.5 mm per jaw; rigid attachments only"
        ),
        "limitations": [
            "Swept bounds cover full interval; unresolved pairs use a discrete sampling grid",
            "Source meshes are visual CAD approximations",
            "Straps, liners, fasteners and external scene excluded",
            "No physical fit or strength validation",
        ],
        "parts": {},
        "parameters_sha256": hashlib.sha256((HERE / "parameters.json").read_bytes()).hexdigest(),
        "mesh_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE / "generated").glob("*.stl")
        },
        "collision_events": [],
    }
    for name, m in parts.items():
        if name.startswith("reference_"):
            continue
        ok = m.is_watertight and m.is_winding_consistent and len(m.split()) == 1 and m.volume > 0
        report["parts"][name] = {
            "watertight": bool(m.is_watertight),
            "consistent_winding": bool(m.is_winding_consistent),
            "components": len(m.split()),
            "positive_volume": bool(m.volume > 0),
        }
        if not ok:
            raise ValueError(f"Bad STL: {name}")
    active = ["right_thumb_lead", "right_finger_lead", "negative_jaw_cap", "positive_jaw_cap"]
    all_meshes = {**ref, **{k: parts[k] for k in active}}
    signs = {
        k: (
            -1
            if k in ("tip_left", "right_thumb_lead", "negative_jaw_cap")
            else 1
            if k in ("tip_right", "right_finger_lead", "positive_jaw_cap")
            else 0
        )
        for k in all_meshes
    }
    pairs = [(a, b) for a in active for b in ref]
    pairs += [(a, b) for i, a in enumerate(active) for b in active[i + 1 :]]
    report["pair_checks"] = []
    for hand in ("right", "left"):
        if hand == "left":
            all_meshes["right_thumb_lead"] = parts["left_finger_lead"]
            all_meshes["right_finger_lead"] = parts["left_thumb_lead"]
        for a, b in pairs:
            relative_sign = signs[a] - signs[b]
            ma, mb = swept_candidates(all_meshes[a], all_meshes[b], relative_sign * TRAVEL)
            record = {
                "hand": hand,
                "a": a,
                "b": b,
                "candidate_faces": [len(ma.faces), len(mb.faces)],
            }
            report["pair_checks"].append(record)
            if not len(ma.faces) or not len(mb.faces):
                record["result"] = "no overlapping swept triangle bounds"
                continue
            collision = vtk.vtkCollisionDetectionFilter()
            collision.SetInputData(0, vtk_poly(ma))
            collision.SetInputData(1, vtk_poly(mb))
            collision.SetCollisionModeToFirstContact()
            collision.SetBoxTolerance(0.0)
            collision.SetCellTolerance(0.0)
            collision.SetNumberOfCellsPerNode(4)
            collision.SetTransform(1, vtk.vtkTransform())
            positions = [0] if relative_sign == 0 else POSITIONS
            record["explicit_samples"] = len(positions)
            record["result"] = "no triangle intersections"
            for q in positions:
                transform = vtk.vtkTransform()
                transform.Translate(relative_sign * q, 0, 0)
                collision.SetTransform(0, transform)
                collision.Update()
                if collision.GetNumberOfContacts():
                    report["collision_events"].append(
                        {"hand": hand, "a": a, "b": b, "q_mm": float(q)}
                    )
                    record["result"] = "intersection"
                    break
        print(f"Completed {hand}-hand rigid clearance sweep", flush=True)
    report["nominal_sampling_grid_count"] = len(POSITIONS)
    report["explicit_vtk_samples"] = sum(
        p.get("explicit_samples", 0) for p in report["pair_checks"]
    )
    report["all_pairs_resolved_by_swept_bounds"] = all(
        p["result"] == "no overlapping swept triangle bounds" for p in report["pair_checks"]
    )
    report["checker_controls"] = checker_controls()
    report["rigid_sweep_passed"] = not report["collision_events"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"valid_stls": len(report["parts"]), "collision_events": report["collision_events"]},
            indent=2,
        )
    )
    return report


def export_scene(items, path):
    s = trimesh.Scene()
    for i, (n, m, c) in enumerate(items):
        m = m.copy()
        m.apply_scale(0.001)
        m.visual.face_colors = [int(c[j : j + 2], 16) for j in (1, 3, 5)] + [255]
        s.add_geometry(m, node_name=f"{n}_{i}", geom_name=f"{n}_{i}")
    s.export(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2rt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    ref = references(args.i2rt)
    parts = load_parts()
    out = HERE / "generated"
    if not args.verify_only:
        board(ref, parts, out / "overview.png")
        export_scene(scene(ref, parts, 0), out / "yam_closed.glb")
        export_scene(scene(ref, parts, TRAVEL), out / "yam_open.glb")
    report = verify(ref, parts, out / "clearance_report.json")
    if not report["rigid_sweep_passed"]:
        raise SystemExit("Rigid surface intersections found; inspect clearance_report.json")
