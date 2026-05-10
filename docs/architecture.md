# Karma architecture

Karma launches a session from YAML. Each node runs in its own process and
communicates over the in-process ZMQ bus through named topics.

```text
leader_left ── leader_left/joint_pos ──▶ left
leader_right ─ leader_right/joint_pos ─▶ right
camera_* ───── camera_*/rgb ───────────▶ policy / visualizer nodes
```

## Core runtime pieces

- `Session` starts/stops nodes, owns recording state, pause/resume, and the TUI.
- `ProcessHost` isolates each node in a subprocess.
- `Node` is the base class for agents, robots, cameras, sim, and visualizers.
- `Publisher` / `Subscriber` move timestamped dictionaries on ZMQ topics.
- Writers record node outputs as MCAP or MP4 under `recordings/`.

## Standard nodes

- `AgentNode` wraps teleop or policy agents and publishes joint commands.
- `RobotNode` wraps YAM robot drivers and publishes `joint_state`.
- `CameraNode` wraps ZED/OpenCV/RealSense camera drivers.
- `XdofSimNode` runs the YAM MuJoCo sim, cameras, Viser, and optional VR stream.
- `ViserMonitorNode` renders YAM URDF overlays, cameras, and policy chunks.

## Recordings

A recording directory contains one MCAP per non-camera node, MP4/timestamp pairs
for cameras, and `session_meta.json` with node wiring and sim metadata.

Typical files:

```text
leader_left.mcap
leader_right.mcap
left.mcap
right.mcap
sim-left.mcap
sim-right.mcap
sim-sim_state.mcap
sim-top-images-rgb.mp4
session_meta.json
```
