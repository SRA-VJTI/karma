# Policy diagnostics and deployment checks

## Before motors

```bash
uv run karma policy doctor --policy-config configs/policies/pi05-yam.json
uv run karma policy probe --policy-config configs/policies/pi05-yam.json \
  --observation observation.npz --instruction "fold the towel"
```

Doctor opens no cameras or native robot interfaces. It resolves the profile and
checks the provider's health/model metadata. Probe performs one model inference
and prints actions without executing them; Dream probe calls can incur API usage.

The NPZ must contain `state` and `top_cam`, plus `left_cam`/`right_cam` for YAM or
`side_cam` for SO101. Arrays are RGB uint8; state is Karma's calibrated canonical
frame. Pickle loading is disabled. Supply `--rig so101` with its normalization
and policy-frame profile for SO101 probes.

## Artifacts

Policy sessions write beneath the existing log root:

```text
runtime/policies/<timestamp>-<id>/
  config.json
  events.jsonl
  dream-worker.log   # Dream only
  sdk/               # official SDK diagnostics
```

`OPENPI_LOG_DIR` overrides the log root. The CLI's existing rotating runtime log
and native crash capture continue to apply. Events contain request/generation
IDs, command anchors, latency, preview acceptance, invalidation, and provider
errors. Images are not included in Karma's event stream. The SDK may produce its
own diagnostics under `sdk/`; handle those as observation data when sharing logs.
Profiles never contain keys. Avoid putting credentials in endpoint URLs.

Rollout/HITL manifests include the resolved policy configuration. Session control
changes do not alter human-intervention labels.

| Symptom | Inspect / action |
| --- | --- |
| Doctor cannot reach `/act` | GPU host, port, firewall, server warmup traceback |
| π0.5 health mismatch | Use Karma's adapter; the older example lacks health metadata |
| Normalization/camera mismatch | Compare loaded checkpoint metadata with profile and rig |
| Wrong SO101 pose | Verify affine policy frame and calibration before motion |
| Dream runtime missing | `uv sync --locked --project runtime/dream`, or set `worker_python` |
| Dream auth/model error | Verify SDK credentials and the exact accessible checkpoint ID |
| Worker deadline exceeded | Inspect `dream-worker.log`; restart the command after fixing the worker |
| Dream pacing becomes sync | Compare request latency with action rate; this can be expected SDK behavior |
| Prediction exhausted during inference | Inspect latency and effective action rate; Resume reobserves |
| Feedback/camera stale | Resolve telemetry or capture delays; Resume requires fresh data |
| Preview differs from motion | Compare model targets with bounds, measured feedback, and progress |
| Stop followed by a late response | Its obsolete generation is discarded; it must never resume motion |

## Validation levels

Automated tests use fake policies and model loaders, without network or motors.
They cover encoding, contract validation, blocked inference, stop/resume,
handoff generations, and command dispatch. An offline passing suite does not
validate GPU compatibility or physical manipulation success.

Dream has also passed a real API probe and a physical recorded YAM rollout.
MolmoAct2 and π0.5 end-to-end checks remain pending for this integration. Track
deployment evidence separately from offline coverage in the
[validation record](validation.md).

Deployment sequence: record versions → start/warm GPU server or SDK runtime →
doctor → saved-observation probe → inspect calibration and camera mapping →
attended low-speed trial. Verify Stop policy holds, Resume requests a fresh chunk,
and Home and stop completes the native shutdown path. Check the predicted
trajectory and measured poses before evaluating task performance.

For explicit payload debugging, set `"capture_payloads": true` in the profile.
The inference worker saves `request-XXXXXX.npz` snapshots and
`actions-XXXXXX.npy` predictions beside events. This is off by default and can
consume substantial storage; recordings may contain private camera imagery.
`versions.json` records the robot-side Python and package versions. Dream's
separate environment is pinned by `runtime/dream/uv.lock`.

`max_prediction_age_s` defaults to 10 seconds. A request that waits through reset
uses the newest available observation and command anchor; a prediction older
than this limit is rejected before execution. This limit is independent of the
network timeout and the existing hardware-feedback freshness checks.
