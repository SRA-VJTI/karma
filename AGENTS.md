# Karma Refactor Notes

This file summarizes the refactor attempt from `robots_realtime` to `karma`, what worked, what broke, and what to preserve when redoing it cleanly.

## Goal

Turn the old Robots Realtime codebase into **Karma**:

> Karma: realtime control and policy runtime for YAM.

Hard scope:

- Yam-only public codebase.
- CLI should be `krm`.
- No `rr-session` / `rr-replay` commands.
- Keep config-driven multi-node runtime.
- Keep cameras for now.
- Keep policy agents for now.
- Keep YAM sim / mjlab for now.
- Remove Franka / Panda / Robotiq support.

## Naming target

Use:

```text
project/package: karma
import path:     karma
CLI:             krm
```

Commands:

```bash
uv run krm session configs/bimanual_yam_leader.yaml
uv run krm replay recordings/<episode_dir>
```

Do **not** create deprecated `rr-session` or `rr-replay` aliases.

## File/package rename

Planned package rename:

```text
robots_realtime/ -> karma/
```

Then globally update imports:

```text
robots_realtime.* -> karma.*
```

Update `pyproject.toml`:

```toml
[project]
name = "karma"
description = "Karma: realtime control and policy runtime for YAM"

[project.scripts]
krm = "karma.cli:main"

[tool.flit.module]
name = "karma"
```

Remove old scripts:

```text
rr-session
rr-replay
```

## CLI shape

Create `karma/cli.py` with subcommands:

```bash
krm session <config.yaml> [--save-root ...] [--no-tui]
krm replay <episode_dir> [...replay args]
```

`krm session` should contain the old `rr_session_cli.py` behavior.

`krm replay` should call package code directly, not shell out to `scripts/replay_episode.py`.

Move:

```text
scripts/replay_episode.py -> karma/replay.py
```

Then `krm replay` imports:

```python
from karma.replay import main as replay_main
```

## Hard-delete non-YAM robot support

Delete these user-facing/code pieces:

```text
configs/franka/
robot_configs/franka/
configs/camera_extrinsics/autolab_franka_zed_top.yaml
media/franka_realtime2.gif
robots_realtime/robots/franka_osc.py
robots_realtime/robots/robotiq_gripper.py
robots_realtime/robots/inverse_kinematics/franka_pyroki.py
robots_realtime/agents/teleoperation/franka_pyroki_viser_agent.py
robots_realtime/agents/teleoperation/franka_pyroki_viser_agent_linear_interp.py
robots_realtime/agents/client/franka_osc_client_cartesian.py
```

Remove Panda dependency pieces:

```text
dependencies/panda_py_0.7.5_libfranka_0.10.0/
```

From `pyproject.toml`, remove:

```toml
franka_panda = ["panda_python"]
panda_python = { path = "..." }
```

Keep for now:

- YAM robot code
- safe motor-chain robot
- YAM PyRoKi IK
- YAM sim / MuJoCo / XDOF sim
- mjlab optional dependency
- ZED/OpenCV/RealSense camera drivers
- ACT/diffusion/OpenPI/LeRobot policy agents/molmo

## Config flattening

Flatten Yam configs aggressively.

Move:

```text
configs/yam/*.yaml -> configs/*.yaml
robot_configs/yam/* -> robot_configs/*
```

Rename config files to remove redundant leading `yam_`:

```text
configs/yam_bimanual_yam_leader.yaml              -> configs/bimanual_yam_leader.yaml
configs/yam_bimanual_gello_teleop.yaml            -> configs/bimanual_gello_teleop.yaml
configs/yam_bimanual_gello_teleop_bair_autolab.yaml -> configs/bimanual_gello_teleop_bair_autolab.yaml
configs/yam_bimanual_passive_gello_teleop_xdof_hq.yaml -> configs/bimanual_passive_gello_teleop_xdof_hq.yaml
configs/yam_bimanual_act_policy_xdof_hq.yaml      -> configs/bimanual_act_policy_xdof_hq.yaml
configs/yam_bimanual_openpi_policy_xdof_hq.yaml   -> configs/bimanual_openpi_policy_xdof_hq.yaml
configs/yam_bimanual_molmoact2.yaml               -> configs/bimanual_molmoact2.yaml
configs/yam_bimanual_obs_viser.yaml               -> configs/bimanual_obs_viser.yaml
configs/yam_single_arm_diffusion_policy_xdof_hq.yaml -> configs/single_arm_diffusion_policy_xdof_hq.yaml
configs/yam_sim_dummy.yaml                        -> configs/sim_dummy.yaml
configs/yam_sim_gello_teleop.yaml                 -> configs/sim_gello_teleop.yaml
configs/README_yam_bimanual_yam_leader.md         -> configs/README_bimanual_yam_leader.md
```

## Topic/node naming convention

Use simple names:

```text
followers: left, right
leaders:   leader_left, leader_right
sim:       sim
```

Replace old names:

```text
yam_left       -> left
yam_right      -> right
yam_leader_left  -> leader_left
yam_leader_right -> leader_right
gello_left     -> leader_left
gello_right    -> leader_right
```

Examples:

```yaml
- type: RobotNode
  name: left
  cmd_topic: leader_left/joint_pos
```

```yaml
state_topics:
  state_left: left/joint_state
  state_right: right/joint_state
```

Sim config node:

```yaml
- type: XdofSimNode
  name: sim
  cmd_topics:
    left: leader_left/joint_pos
    right: leader_right/joint_pos
```

## Runtime/temp naming

Replace:

```text
rr_logs_ -> karma_logs_
rr_vr_   -> karma_vr_
```

TUI panel title:

```text
robots_realtime -> karma
```

## Media rename

Renamed during attempt:

```text
media/real_yams_rr.gif -> media/real_yams_karma.gif
media/rr_vr_support.gif -> media/karma_vr_support.gif
```

This is optional but consistent.

## Docs cleanup

Rewrite docs around Karma/YAM only:
also make a agents.md file in root dir for agents to have first citizen support

```text
README.md
docs/architecture.md
docs/extending.md
docs/vr_streaming.md
configs/README_bimanual_yam_leader.md
```

Remove stale references to:

```text
robots_realtime
rr-session
rr-replay
configs/yam
robot_configs/yam
Franka
Robotiq
Panda
```

README should include:

- `Karma: realtime control and policy runtime for YAM`
- install instructions (uv only1)
- `krm session`
- `krm replay`
- config list
- topic naming conventions



any how the molmo agent code should work !!!!
