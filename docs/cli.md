# Karma commands

The four main workflows are:

```bash
uv run karma teleop --help
uv run karma teleop --record --help
uv run karma inference --help
uv run karma rollout --help
uv run karma hitl --help
```

Maintenance commands are `calibrate-so101`, `doctor`, `zero`, `cameras`, and
`relay`. `live`, `record`, and `collect` remain lower-level utilities; `infer`
is a compatibility alias for `inference`. The executable is `karma`.

Common hardware flags are `--rig`, repeated `--interface ARM=BUS`, and repeated
`--calibration ARM=FILE` for SO101. Camera-enabled workflows accept repeated
`--camera-serial NAME=SERIAL`. Policy workflows accept `--server` and `--norm-tag`.
Run a subcommand's help for its complete flag set. See the [README](../README.md)
for full examples and the [camera guide](cameras.md) for role configuration.

## Zero

`karma zero` writes firmware zeros. Preview with `--dry-run`. For YAM, use only
the manufacturer's mechanical reference pose; see [YAM setup](yam-setup.md).
For SO101 use the dedicated calibration wizard, which backs up firmware registers
and records travel, gripper direction, and home. A home pose is not a firmware
zero. Never replace calibration with zeroing at an arbitrary pose.
