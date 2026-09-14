# RealSense cameras

Karma captures RGB with the RealSense SDK. Use USB 3, rigid mounts and serial
numbers: `/dev/videoN` changes when devices reconnect. The default capture is
848×480 at 30 fps; confirm the camera offers this mode. D405 wrist cameras and
D435 overhead cameras have been used on this workstation. Keep camera framing,
rotation, exposure and lighting consistent between demonstrations and deployment.

Discover devices with the installed SDK:

```bash
uv run python -c 'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
```

## SO101

Only `top` is declared by default. Its placeholder serial is deliberately not a
real device: supply `--camera-serial top=YOUR_SERIAL` on recording/policy commands.
Wrists are optional; supplying their serial adds the view:

```bash
uv run karma cameras --rig so101_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_SERIAL \
  --camera-serial right_wrist=RIGHT_SERIAL --probe
```

A single-arm rig accepts only its own wrist role. Plain teleop can run without
cameras. Recording can explicitly use `--no-cameras` for state-only diagnostics;
policy inference requires at least the top view.

## YAM

The existing YAM checkpoint uses top, left wrist and right wrist in that order.
Configure all three for your own hardware; the packaged serials describe the
original workstation, not every installation:

```bash
uv run karma cameras --rig yam_bimanual \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_SERIAL \
  --camera-serial right_wrist=RIGHT_SERIAL --probe
```

Pass these same flags to `teleop --record`, `inference`, `rollout`, and `hitl`.
`--camera NAME=DEVICE` is a lower-level device-path override. `--camera-serial`
changes the rig's stable camera identity. Do not mix the two unless diagnosing
discovery. A missing declared policy camera fails startup.

The relay and recorder share capture readers during collection/HITL. Close
RealSense Viewer and other camera processes first. For camera mode/USB failures,
check the connection and the SDK's supported profiles rather than substituting
a missing wrist with the top image: that changes the checkpoint's observation.

Camera extrinsics only affect visualization metadata. Policy input is RGB;
there is no automatic hand-eye calibration or image-space correction. Keep the
physical placement used by your training data. Optional rotation/resolution
settings live in the rig's `RigCamera` definition.

## Local configuration

Packaged camera serials are synthetic placeholders. Pass `--camera-serial
ROLE=SERIAL` for each physical camera as shown in the README. Keep local camera
maps and calibration files under the ignored `calibration/` directory.
The standalone relay accepts `CAM_MAP=calibration/camera-map.json`, a JSON
object mapping serial numbers to `top`, `left`, or `right` roles.
The default overhead camera pose is an illustrative view one metre above the
arm midpoint; it is not a measured calibration of your workstation.
