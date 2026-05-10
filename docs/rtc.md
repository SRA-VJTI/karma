# OpenPI RTC notes

Karma's OpenPI policy client lives at
`karma.agents.policy_learning.async_pi0_agent:AsyncDiffusionAgent` and is wired
by `configs/bimanual_openpi_policy_xdof_hq.yaml`.

Run a remote OpenPI server, then launch:

```bash
uv run krm session configs/bimanual_openpi_policy_xdof_hq.yaml
```

The policy node subscribes to:

```yaml
state_topics:
  left: left/joint_state
  right: right/joint_state
image_topics:
  left_camera: camera_left/rgb
  right_camera: camera_right/rgb
  top_camera: camera_top/rgb
```

It publishes `openpi_policy/left_pos`, `openpi_policy/right_pos`, optional
`openpi_policy/chunk`, and preprocessed image snapshots under
`openpi_policy/image/*` for visualization.
