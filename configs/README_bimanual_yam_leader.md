# Bimanual YAM leader config

`configs/bimanual_yam_leader.yaml` runs two YAM teaching-handle leader arms
against two YAM follower arms.

## Hardware

- `leader_left` on `robot_configs/leader_left.yaml`
- `leader_right` on `robot_configs/leader_right.yaml`
- `left` on `robot_configs/left.yaml`
- `right` on `robot_configs/right.yaml`

Bring up the CAN interfaces first:

```bash
./scripts/setup_can_yam_bimanual_leader.sh
```

Then launch:

```bash
uv run krm session configs/bimanual_yam_leader.yaml
```

## Topics

- `leader_left/joint_pos` commands `left`.
- `leader_right/joint_pos` commands `right`.
- `left/joint_state` and `right/joint_state` feed bilateral leader feedback.
- `leader_left/record` toggles recording.
