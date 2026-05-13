# Karma

**Karma: realtime control and policy runtime for YAM.**

Karma runs config-driven, multi-process realtime sessions for YAM follower arms,
YAM/GELLO-style leader arms, cameras, simulation, and policy agents.

## Install

```bash
git clone --recurse-submodules https://github.com/SRA-VJTI/karma.git
cd karma
uv sync
```

Optional camera/sim extras:

```bash
uv sync --extra sensors
uv sync --extra realsense
uv sync --extra mjlab-sim
```

## Run

```bash
uv run krm session configs/bimanual_yam_leader.yaml
uv run krm session configs/sim_gello_teleop.yaml
uv run krm session configs/bimanual_openpi_policy_xdof_hq.yaml --save-root recordings
```

Disable the TUI if needed:

```bash
uv run krm session configs/sim_dummy.yaml --no-tui
```

Replay a recorded sim episode:

```bash
uv run krm replay recordings/<date>/<episode_dir>
```

## Configs

- `configs/bimanual_yam_leader.yaml` — YAM active leader arms → YAM followers.
- `configs/bimanual_gello_teleop.yaml` — serial GELLO leaders → YAM followers.
- `configs/bimanual_gello_teleop_bair_autolab.yaml` — lab camera variant.
- `configs/bimanual_passive_gello_teleop_xdof_hq.yaml` — SocketCAN passive leaders.
- `configs/bimanual_act_policy_xdof_hq.yaml` — remote ACT policy deployment.
- `configs/bimanual_openpi_policy_xdof_hq.yaml` — OpenPI policy deployment.
- `configs/bimanual_molmoact2.yaml` — MolmoAct2 policy deployment.
- `configs/sim_dummy.yaml` — autonomous dummy leaders into the YAM sim.
- `configs/sim_gello_teleop.yaml` — leader teleop into the YAM sim.

Robot hardware configs live directly under `robot_configs/`.

## Naming conventions

Use simple bus node names:

- followers: `left`, `right`
- leaders: `leader_left`, `leader_right`
- simulation: `sim`

Examples:

```yaml
- type: RobotNode
  name: left
  cmd_topic: leader_left/joint_pos
```

```yaml
state_topics:
  left: left/joint_state
  right: right/joint_state
```

```yaml
- type: XdofSimNode
  name: sim
  cmd_topics:
    left: leader_left/joint_pos
    right: leader_right/joint_pos
```
