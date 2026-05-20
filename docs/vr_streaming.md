# VR streaming

`XdofSimNode` can stream the YAM MuJoCo scene to a Quest/Meta headset through a
small Three.js WebXR server.

## Run

```bash
uv run krm session configs/sim_gello_teleop.yaml
```

If a Quest is connected over ADB, Karma exports scene meshes to `/tmp/karma_vr_*`,
sets up reverse port forwarding, and opens the headset browser. The desktop Viser
viewer remains available at the configured `viser_port`.

Disable VR for a sim config by setting:

```yaml
vr_port: null
```

## Notes

- Install and authorize `adb` on the host machine.
- Keep `viser_port` and `vr_port` distinct.
- VR streaming is visualization-only; control still flows through normal Karma
  topics such as `leader_left/joint_pos`.
