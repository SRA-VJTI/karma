"""Measure what a normalized gripper command actually does on this hardware.

Our E_Yam config asserts that normalized [0, 1] spans [0.0, 4.5] relative
radians of the gripper servo. i2rt -- which recorded the MolmoAct2 training
data -- refuses to assert it: ``linear_4310.yml`` ships
``gripper_limits: null, needs_calibration: true`` and calibrates the closed and
open stops at every startup, describing the same hardware's stroke as 6.57 rad.

So this walks the gripper down its normalized range and prints what the arm
reports back, which answers the two questions a policy cares about:

  * does normalized 0.0 actually close the gripper, or stop short of the stop?
  * what range does the arm report, and is it the [0, ~0.98] the checkpoint was
    trained on (norm_stats q01 0.05, q90 0.98)?

Arm joints are held exactly where they are; only the gripper moves.

    uv run python scripts/gripper_range.py --arm left
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from openpi_control.cli import power_down, power_up
from openpi_control.rigs import resolve_rig
from openpi_control.types import PositionCommand

# The E_Yam catalog sets need_repeated_command, so a target has to be held by
# resending it rather than sent once.
_HOLD_S = 2.0
_RATE_HZ = 30.0
_TARGETS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="left", help="which arm's gripper (default: left)")
    parser.add_argument("--rig", default="yam_bimanual")
    parser.add_argument(
        "--hold", type=float, default=_HOLD_S, help="seconds to hold each target"
    )
    args = parser.parse_args(argv)

    rig = resolve_rig(args.rig).subset([args.arm])
    session, live_arms = power_up(rig, float_mode=False)
    entry = live_arms[0]
    arm = entry.arm
    try:
        state = arm.latest_state
        if state is None or state.effector is None:
            print(f"{entry.name}: no gripper state; is the E_Yam configured?", file=sys.stderr)
            return 1
        held = np.asarray(state.joints.position_rad, dtype=np.float64)
        print(f"holding {entry.name} joints at {np.round(held, 3).tolist()}")
        print(f"starting gripper reading: {state.effector.position:.4f}\n")
        print(f"{'commanded':>10}  {'measured':>9}  {'error':>7}  what you should see")
        print("-" * 62)

        period = 1.0 / _RATE_HZ
        results = []
        for target in _TARGETS:
            deadline = time.monotonic() + args.hold
            while time.monotonic() < deadline:
                arm.command(PositionCommand(held, effector=target))
                time.sleep(period)
            reached = arm.latest_state
            measured = reached.effector.position if reached and reached.effector else float("nan")
            results.append((target, measured))
            print(f"{target:10.2f}  {measured:9.4f}  {measured - target:+7.4f}")

        spread = [m for _, m in results if not np.isnan(m)]
        print("-" * 62)
        if spread:
            print(f"reported range over the sweep: {min(spread):.4f} .. {max(spread):.4f}")
            print(
                "the checkpoint was trained on 0.05 (closed) .. 0.98 (open); "
                "a much narrower range here means our [0, 4.5] rad assumption is wrong"
            )
        return 0
    finally:
        power_down(session, live_arms, park=True)


if __name__ == "__main__":
    raise SystemExit(main())
