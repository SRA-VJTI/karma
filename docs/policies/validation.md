# Policy integration validation

Last updated: 2026-11-08. This record covers Karma's shared policy integration
and maintained serving adapters. Support in code and offline tests is distinct
from an end-to-end deployment check.

| Provider | API/model prediction | Physical recorded rollout | Status |
| --- | --- | --- | --- |
| Dream, `dreamlabs/dream-1-beta` | Doctor and synthetic-observation probe passed | Bimanual YAM, 120 seconds, 3,593 frames; operator marked successful | Tested; remaining control-path checks below |
| MolmoAct2 | Maintained server and shared client end-to-end check pending | Pending | Next to validate |
| π0.5, `robocurve/pi0.5-yam` | Maintained server and shared client end-to-end check pending | Pending | Next to validate |

## Dream: completed checks

- Isolated SDK runtime installed; doctor returned `health.status: ok` for
  `dreamlabs/dream-1-beta`.
- A synthetic-observation probe returned numeric action chunks without opening
  motors. This verified API prediction, not camera mapping or task performance.
- Physical `karma rollout` ran on `yam_bimanual` with top, left-wrist, and
  right-wrist cameras, prompt **"fold the towel"**, `--speed 1`, one episode,
  and `--episode-seconds 120`.
- Startup waited for fresh feedback and began policy execution automatically
  once observations were available. No initial Resume press was required.
- Viser connected on port 8080 during the rollout.
- The episode saved 3,593 frames. Both arms parked at `home_pos` and
  de-energized on normal completion. The operator answered `y` to episode success.
- A camera collage MP4 was generated and stored locally with the task prompt,
  overhead view, and both wrist views.

Local evidence (recordings and runtime artifacts are not committed):

| Artifact | Location |
| --- | --- |
| Dataset ID | `local/dream-towel-rollout-02` |
| Dataset directory | `~/.cache/huggingface/lerobot/local/dream-towel-rollout-02` |
| Policy run | `logs/runtime/policies/20261108T231202-7d8ac732/` |
| Video collage | `outputs/rollout-collages/dream-towel-rollout-02-collage.mp4` |
| Local profile | `configs/policies/dream-yam.local.json` |

The tested local profile selects `dreamlabs/dream-1-beta`; the checked-in example
selects `dreamlabs/dream-1`. These are separate checkpoints. Raw prediction payload
capture was disabled for this run: response metadata and recorded executed actions
are available, but the complete returned chunks were not saved. The camera collage
does not display raw Dream predictions.

The initial inference trial exposed a missing runtime `PositionCommand` import;
that was fixed. Startup feedback recovery was also fixed before the successful
rollout. Neither that earlier trial nor the successful rollout establishes a
separate completed validation of the standalone inference command.

## Remaining deployment checks

Run the following for **each of MolmoAct2 and π0.5**, using their respective
[MolmoAct2](molmoact2.md) and [π0.5](pi05.md) setup guides:

- [ ] Load the intended checkpoint on the GPU host and complete server warmup;
  record checkpoint, dependencies, GPU, and server command.
- [ ] Run `karma policy doctor` against the server and verify health metadata.
- [ ] Run a saved-observation probe; check action shape, joint/gripper ordering,
  units, normalization, and camera roles against the trained model contract.
- [ ] Run attended low-speed inference; inspect predicted chunks in Viser and
  compare them with measured robot motion.
- [ ] Verify Stop policy, Resume policy with a fresh prediction, and Home and stop
  on hardware. Confirm a late response cannot restart stopped motion.
- [ ] Record a short rollout; verify camera synchronization, saved actions,
  episode metadata, and normal parking/de-energization.
- [ ] Exercise HITL takeover and policy handback; confirm obsolete predictions
  are discarded and resumption uses fresh observations.
- [ ] Attach the run artifact path, dataset ID, and operator outcome here.

For MolmoAct2, validate SO101 separately if deploying that profile; the YAM trial
does not establish SO101 calibration or policy-frame correctness.

Remaining Dream checks:

- [ ] Complete a standalone `karma inference` trial after the runtime import fix.
- [ ] Explicitly exercise Viser Stop/Resume/Home on hardware; a connected viewer
  and successful normal shutdown do not establish that its buttons were tested.
- [ ] Exercise physical HITL takeover and handback.
- [ ] Validate other checkpoints or joint-only profiles separately if used.

Automated tests cover shared scheduling, invalidation, worker behavior, serving
contracts, and viewer controls using fake policies/models. They complement these
deployment checks and do not substitute for them.
