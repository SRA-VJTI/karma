# MolmoAct2 integration

Karma is the robot-side client. Run your MolmoAct2 model server on a GPU host;
Karma connects to `http://HOST:8202/act` using JSON with `json_numpy` arrays.
Checkpoint training/loading remains in the model environment. There is no
assumption that a YAM checkpoint knows how to move SO101.

## Contract

| Rig | State/action width | Joint order per arm | Views | Gripper on policy wire |
| --- | --- | --- | --- | --- |
| `yam_bimanual` | 14 | 6 joint radians, gripper | top, left wrist, right wrist | 1=open |
| `so101` | 6 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | top; right wrist optional | 0=open |
| `so101_bimanual` | 12 | same SO101 block, left then right | top; wrists optional | 0=open |

Arm joints are absolute **calibrated radians**, not degrees or LeRobot's
percentage-normalized servo positions. Actions are absolute joint targets,
not deltas. SO101 uses the same 0=open gripper convention as Karma's datasets;
the original YAM HTTP contract retains 1=open and needs that conversion in its
training/server adapter. Native hardware always uses 1=open. Do not apply this
conversion a second time in a SO101 server.

SO101 requires `--norm-tag YOUR_CHECKPOINT_KEY`. Get it from the model's actual
normalization metadata; there is no universal SO101 tag. The client checks server
health before powering motors. Non-default contracts require these fields:

```json
{"status":"ok", "norm_tag":"YOUR_CHECKPOINT_KEY", "state_dim":6, "num_cameras":1}
```

For bimanual SO101 use `state_dim:12`; adding wrists changes `num_cameras`.
The server must train and serve those views in top/left/right order. Karma never
pads missing joints or duplicates a missing camera to satisfy a YAM checkpoint.

A POST carries `state`, `instruction`, `timestamp`, `normalization_tag`,
`num_steps`, `enable_cuda_graph`, and the configured image keys `top_cam`,
`left_cam`, `right_cam`. Images are JPEG byte arrays encoded with `json_numpy`
by default, or H×W×3 RGB arrays with `--raw-frames`. Return
`{"actions": <N by state_dim array>}` using the same codec. Empty, wrong-width
or non-finite action chunks are rejected.

SO101 commands use calibrated joint limits and bounded per-tick changes.
The legacy YAM `--reach-actions` and demonstration start-pose ramp do not apply
to SO101. Keep the same control cadence, camera preprocessing, units and
normalization used during training. `--speed` slows chunk playback;
`--max-step-rad` bounds a command increment.

See the [README](../README.md) for executable inference/rollout/HITL examples.
The tests exercise both arm layouts with simulated states and HTTP responses;
they do not establish that a particular checkpoint performs a physical task.
