# Viser visualization

`openpi_control.viz` serves a packaged arm model in the browser. It resolves the
same URDF and model JSON that `openpi_control.config` hands to the native node,
and draws them over HTTP.

It opens no CAN, serial, Ethernet, or Modbus connection and starts no
`pi_control_node`, so it runs with the buses down and no arm attached.

## Install

Its dependencies are an optional extra -- nothing here is needed to drive an
arm -- but they are in the default `cell` group, so a plain sync installs them:

```bash
uv sync
```

## Getting the real arm

The wheel ships each URDF but not its meshes — the URDFs are here for the
gravity-compensation model, which needs link inertias and joint origins and
never needs geometry. Fetch the visual meshes once:

```bash
uv run openpi-control-viz --fetch-meshes --model Yam
```

That downloads I2RT's YAM meshes (MIT) into `~/openpi-data/meshes/Yam/`, beside
the run logs. Every later run finds them automatically and needs no network, so
the real arm is the default with no flags. Provenance, including the pinned
upstream revision, is written to `SOURCE.txt` next to the meshes.

i2rt keeps the wrist geometry with its crank gripper rather than with the arm,
so `link_6_visual.stl` / `link_6_collision.stl` are not in the assets directory
the other twelve meshes come from — both are fetched from the gripper model
instead. A mesh that is genuinely absent is reported on startup and in the GUI
rather than left silent, and that link renders bare.

## Standalone viewer

GUI sliders drive the joints:

```bash
uv run openpi-control-viz --model Yam --effector E_Yam    # one arm
uv run openpi-control-viz --list                          # models with a URDF
```

Then open <http://localhost:8080>.

| Flag | Meaning |
| --- | --- |
| `--model` | arm model (default `Yam`) |
| `--effector` | effector model, shown in the GUI summary |
| `--instance-config` | override the per-unit instance JSON |
| `--urdf` | override the packaged URDF (required for FR3) |
| `--mesh-dir` | directory holding the URDF's meshes; beats the cache |
| `--fetch-meshes` | download `--model`'s meshes into the cache, then exit |
| `--port` | HTTP port (default 8080) |
| `--no-grid` | hide the ground grid |
| `--list` | list which models ship a URDF, then exit |
| `--rig` | draw a whole packaged rig instead of one arm |
| `--list-rigs` | list the packaged rigs and their arms, then exit |

## Render modes

**`skeleton`** (the default) draws a coordinate frame per link, a bone between
each link origin and its parent's, and an orange marker on each actuated joint
axis. It needs no mesh files.

**`mesh`** draws the URDF's visual geometry via `viser.extras.ViserUrdf`.

The mesh directory is resolved in this order: an explicit `--mesh-dir`, then a
sibling `assets/` beside the URDF, then `~/openpi-data/meshes/<Model>/` — the
cache `--fetch-meshes` fills. With none of those, you get the skeleton, which is
why a fresh checkout renders one.

Only `Yam` has a known upstream mesh source today; other models need
`--mesh-dir` pointed at meshes you already have.

## Bimanual and other rigs

A rig draws a whole cell in one scene, each arm at its own base pose. Still
hardware-free — the sliders drive the render:

```bash
uv run openpi-control-viz --list-rigs
uv run openpi-control-viz --rig yam_bimanual
```

`--rig` takes its models from the rig, so it refuses to be combined with
`--model`. From Python:

```python
from openpi_control.rigs import resolve_rig
from openpi_control.viz import ArmSceneVisualizer

scene = ArmSceneVisualizer.from_rig(resolve_rig("yam_bimanual"))
scene.update("left", left.read_state().joints.position_rad)
```

The rig owns the arm names, so `"left"` is the same string the operator CLI and
the session use for that arm. Each arm gets its own tint so a two-arm scene
stays readable, and `ArmSpec` is still there for a scene the packaged rigs do
not cover.

The packaged `yam_bimanual` scene also shows the supplied calibrated `top`
camera frame. Its pose is the camera-to-midpoint transform from
`openpi_control.camera_poses`; the inverse and RPY values are derived from the
same rigid transform. The wrist-camera frames are intentionally not drawn
until their extrinsics are calibrated.

To mirror two *live* arms rather than sliders, use `openpi-control live` — it
owns the power-on and power-off that a live view implies. See
[docs/cli.md](cli.md#live).

## Colour and theme

Every page this package serves asks viser for **dark mode**, because every
colour in the scene was picked against a dark canvas and several of them do not
survive the white one viser serves by default:

| | on dark | on viser's default white |
| --- | --- | --- |
| amber axis markers | 8.6:1 | **2.0:1** |
| camera-tile placeholder | 1.1:1 (recedes, as intended) | **15.9:1** (a black hole in a white page) |
| left arm, blue `(92,132,186)` | 4.5:1 | 3.8:1 |
| right arm, amber `(238,172,86)` | 8.8:1 | 2.3:1 |

The two arm tints are blue and amber — the pair that survives the common
red-green colour-blindnesses — and they are separated by **lightness** as well
as hue, landing 66/255 apart in greyscale. That matters because the predicted
chunk trails are translucent and overlap: hue alone stops being a distinction
where two trails cross, and disappears entirely in a greyscale screenshot. The
earlier amber was 13/255 from the blue, which is one colour to anyone not seeing
hue.

The GUI accent is a teal `(96,165,168)` that is deliberately *not* an arm tint,
so a highlighted slider never reads as "the left arm". The share button is off:
it publishes the page through viser's relay, and these pages show a live robot
cell.

Only pages this package creates are themed. Hand `ArmVisualizer` or
`ArmSceneVisualizer` your own `server=` and its chrome stays yours.

## Driving it from a live arm

`ArmVisualizer` only draws — the caller owns the hardware session, so live
visualization stays an explicit opt-in:

```python
from openpi_control import ArmConfig, ArmSession, SocketCanConnection
from openpi_control.viz import ArmVisualizer

viz = ArmVisualizer("Yam", effector_model="E_Yam")
print(viz.url)

with ArmSession() as session:
    follower = session.add_follower(
        ArmConfig("right_follower", "Yam", SocketCanConnection("can_follower_r"),
                  effector_model="E_Yam")
    )
    session.connect()
    while True:
        viz.update(follower.read_state().joints.position_rad)
```

`update()` takes radians as a sequence in joint-index order, or as a
`{joint_name: value}` mapping that may be partial. Values are clamped to the
URDF limits, so a stale or out-of-range sample cannot fold the render through a
joint stop.

## Joint ordering

`ArmVisualizer.joint_names` is ordered by depth from the base link, matching how
`PositionCommand` and `ArmState` index joints.

This is deliberately not `yourdfpy`'s `actuated_joint_names`, which follows URDF
document order — and `Yam.urdf` and `SO101.urdf` both declare their joints
tip-first. Indexing by document order would drive joint 1 from joint 6's
command. In mesh mode the positions are permuted back into `ViserUrdf`'s own
order before rendering.

## Models

`FR3` ships no URDF; its kinematics live in the vendor controller. Pass
`--urdf` to visualize it. Every other model in `SUPPORTED_MODELS` renders from
its packaged URDF.
