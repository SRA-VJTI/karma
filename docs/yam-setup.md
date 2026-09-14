# YAM setup

YAM uses six arm motors and an E_Yam gripper on each SocketCAN bus. The default
bimanual rig is left `can0`, right `can1`. Inspect adapter names first:

```bash
ip -brief link show type can
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
uv run karma doctor --rig yam_bimanual
```

Use `--interface left=canN --interface right=canM` on runtime commands if your
wiring differs. Never run two robot processes on the same bus.

## Zeros and home

YAM uses firmware zeros and the packaged native instance configuration, not a
LeRobot SO101 calibration file. Only write zeros with the arm positioned in the
manufacturer's documented mechanical zero pose. A chosen resting/home pose is
not necessarily the mechanical zero pose.

```bash
uv run karma zero --model Yam --interface can0 --effector E_Yam --dry-run
# After placing the arm at the correct mechanical reference:
uv run karma zero --model Yam --interface can0 --effector E_Yam
```

Repeat for the other arm only if its firmware calibration needs correction.
`src/openpi_control/models/arms/Yam/Yam_01.json` stores per-servo `zero_pos` and
`home_pos` in radians; `home_pos` is relative to the calibrated joint zero.
The effector instance is `models/effectors/E_Yam/E_Yam_01.json`. Review these
values before enabling automatic parking. `--rest-pose-left` / `--rest-pose-right`
on teleop override the Quest thumbstick rest target, not firmware calibration.

Start `uv run karma teleop --rig yam_bimanual --open-quest`. Verify measured
joint directions and gripper jaws with small clutched movements. Native gripper
position is 1=open and 0=closed. See [cameras](cameras.md) for RealSense setup.
The optional passive teaching handle remains supported; see
[yam_teaching_handle.md](yam_teaching_handle.md).
