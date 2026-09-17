# LeRobot v3 recording

All persisted demonstrations (`karma teleop --record`), policy rollouts
(`karma rollout`) and DAgger episodes (`karma hitl`) use the same LeRobot sink.
The dependency is pinned to LeRobot 0.6.1 and the writer checks `CODEBASE_VERSION`
is `v3.0`. There is no v2 output mode.

```bash
uv run karma teleop --record --rig so101 \
  --interface right=/dev/ttyACM0 --calibration right=calibration/so101-right.json \
  --camera top=/dev/video5 --camera side=/dev/video7 --repo-id local/so101-demo \
  --task "pick up the object" --fps 30 --open-quest
```

Cameras are RealSense (`--camera-serial ROLE=SERIAL`) or USB webcams
(`--camera ROLE=/dev/videoN`); see [cameras](cameras.md).

`--root PATH` chooses the dataset directory; otherwise LeRobot uses its dataset
cache. Choose a new dataset ID/root for a new run. Upload is opt-in with
`--push-to-hub` and optionally `--private`; authenticate with Hugging Face first.

## Schema

- `observation.state`: measured arm joints in calibrated radians plus normalized
  gripper, in rig order (left then right for bimanual). Rollout/HITL datasets are
  in this frame too; a `--policy-frame` only touches the HTTP request.
- `action`: commanded targets using the same joint order and units.
- Dataset gripper convention: **0=open, 1=closed** for every robot.
- `observation.images.NAME`: RGB video for each configured camera role.
- Task text and LeRobot frame/episode timestamps and indexes.
- HITL includes `intervention` for human control frames.

SO101 vectors have 6 or 12 entries; bimanual YAM vectors have 14. Do not mix
embodiments or camera schemas in one dataset. LeRobot v3 stores chunked parquet
and video with metadata under `meta/`; do not assume one file per episode.

The recorder skips stale hardware observations and does not turn repeated stale
samples into demonstrations. RGB readers are shared with the Quest relay during
collection and HITL. Episodes are reviewed/saved through the workflow's controls;
discarded attempts do not become training episodes. Streaming video encoding
avoids staging every camera frame as a temporary image.

Rollout and HITL also write run manifests with task/outcome information. In DAgger,
train corrections from intervention-labeled frames according to your training
recipe; the label means who commanded the action, not whether the episode was
successful. See [DAgger workflow](dagger.md).
