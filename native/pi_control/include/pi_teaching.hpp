#pragma once

#include <algorithm>
#include <cmath>

// Joint-relative Nm before the existing torq_rescale conversion. No constant
// torque at rest: the continuous sign transition avoids a stiction kick.
struct TeachingJoint {
    float coulomb = 0.0f;
    float viscous = 0.0f;
    float transition = 0.05f;
    float compensation_limit = 0.0f;
    float velocity_limit = 1.0f;
    float damping = 0.1f;
    bool configured = false;

    bool valid(float max_compensation) const {
        return std::isfinite(coulomb) && std::isfinite(viscous) &&
               std::isfinite(transition) && std::isfinite(compensation_limit) &&
               std::isfinite(velocity_limit) && std::isfinite(damping) &&
               coulomb >= 0 && viscous >= 0 && transition >= 0.01f &&
               compensation_limit >= 0 && compensation_limit <= max_compensation &&
               coulomb <= compensation_limit && viscous <= max_compensation &&
               velocity_limit > 0 && velocity_limit <= 3.0f && damping >= 0 && damping <= 5.0f;
    }

    float torque(float velocity) const {
        const float assist = coulomb * std::tanh(velocity / transition) + viscous * velocity;
        return std::clamp(assist, -compensation_limit, compensation_limit) - damping * velocity;
    }
};
