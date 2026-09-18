# HITL / DAgger workflow

1. Calibrate each arm and configure the cameras used by the checkpoint.
2. Collect initial demonstrations with `karma teleop --record`.
3. Train or fine-tune MolmoAct2 in its training environment with the documented
   joint units, camera roles and dataset gripper convention.
4. Serve that checkpoint and verify its health contract ([model servers](model-servers.md));
   use `karma inference` for a supervised first run or `karma rollout` for timed
   recorded attempts. A checkpoint trained in another joint frame needs
   `--policy-frame` ([inference](inference.md#policy-frames)).
5. Run `karma hitl`. The policy drives until **Right B** takes control.
6. Grip to clutch and correct the trajectory. Tap **Left Y** to hand back;
   the policy replans from the measured corrected state. Hold Left Y to end
   the attempt. Follow terminal review prompts to accept, label, or discard.
7. Aggregate accepted data, select intervention labels/outcomes for your recipe,
   retrain, and repeat using a new dataset/checkpoint version.

Single right-arm SO101 still uses the left Quest controller's Y button for
handoff/end; keep both controllers available. Mode switches discard stale policy
plans and reseed command targets, including the last commanded gripper state.
The shared LeRobot v3 writer records policy and human ticks in one episode with
an intervention label, always in Karma's radians regardless of any policy frame. Training, evaluation criteria and GPU model serving are
separate processes, not implicit side effects of collecting data.
