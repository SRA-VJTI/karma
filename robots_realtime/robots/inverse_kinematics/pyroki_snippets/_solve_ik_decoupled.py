"""
Pieper-style decoupled IK with manipulability cost and warm-starting.

Stage 1: position-only solve (pos_weight=200, ori_weight=0) — drives shoulder/elbow
         to the right place without caring about orientation.
         A rest_cost anchored to the previous solution prevents elbow flips.
         A manipulability_cost steers away from singularities.

Stage 2: full pose solve with a strong joint anchor to Stage 1 — the arm posture is
         already good so only wrist joints move to hit the orientation target.
         Manipulability cost retained to avoid wrist singularities.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_decoupled(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    prev_joints: Optional[onp.ndarray] = None,
) -> onp.ndarray:
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    link_idx = jnp.array(robot.links.names.index(target_link_name))
    wxyz = jnp.array(target_wxyz)
    pos = jnp.array(target_position)

    # Warm-start: anchor stage 1 to previous solution to prevent elbow flips.
    # Falls back to zero config on first call.
    warm = (
        jnp.array(prev_joints)
        if prev_joints is not None
        else jnp.zeros(robot.joints.num_actuated_joints)
    )

    stage1 = _stage1(robot, link_idx, wxyz, pos, warm)
    stage2 = _stage2(robot, link_idx, wxyz, pos, stage1)
    return onp.array(stage2)


@jdc.jit
def _stage1(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    warm_start: jax.Array,
) -> jax.Array:
    """Position-only: drives arm posture to the right place.
    warm_start anchors to previous solution → prevents elbow flips.
    manipulability_cost steers away from singular configurations.
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=200.0,
            ori_weight=0.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
        # Continuity: anchor posture to previous solution. Weight is moderate —
        # strong enough to prevent elbow flips, weak enough to allow movement.
        pk.costs.rest_cost(joint_var, rest_pose=warm_start, weight=15.0),
        # Singularity avoidance: penalise 1/manipulability (Yoshikawa measure).
        pk.costs.manipulability_cost(robot, joint_var, target_link_index, weight=0.05),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


@jdc.jit
def _stage2(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    stage1_joints: jax.Array,
) -> jax.Array:
    """Orientation refinement.
    The strong rest_cost anchors the arm to Stage 1 so the optimizer mainly
    moves wrist joints to fix orientation.  Manipulability cost prevents
    wrist singularities.
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=20.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
        # Anchor arm posture to Stage 1 — wrist joints have lower Jacobian
        # contribution to position so they carry the orientation residual.
        pk.costs.rest_cost(joint_var, rest_pose=stage1_joints, weight=80.0),
        pk.costs.manipulability_cost(robot, joint_var, target_link_index, weight=0.05),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
