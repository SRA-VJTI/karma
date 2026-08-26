/*!
 * @file test_servo_turn_tracking.cpp
 * @brief Continuous multi-turn accumulation of single-turn position feedback.
 *
 * The case that matters is a gripper whose stroke is longer than one feedback
 * turn (the linear_4310: i2rt measures 6.57 rad against the DM4310's 6.283 rad
 * period). Its two ends alias onto nearly the same reading, so the startup
 * one-shot wrap has no unique answer and the accumulator has to carry the turn.
 */

#include <gtest/gtest.h>

#include <cmath>

namespace {

// The accumulator under test, kept in lockstep with Servo::accumulate_position.
// Servo itself is abstract and drags in the driver/device graph; the arithmetic
// is what has to be right, so it is exercised directly.
class TurnAccumulator {
   public:
    explicit TurnAccumulator(float period) : period_(period) {}

    void accumulate(float raw_abs) {
        if (period_ <= 0) {
            value_ = raw_abs;
            return;
        }
        if (!started_) {
            started_ = true;
            last_raw_ = raw_abs;
            value_ = raw_abs;
            return;
        }
        float delta = raw_abs - last_raw_;
        const float half = period_ * 0.5f;
        if (delta > half) {
            delta -= period_;
        } else if (delta < -half) {
            delta += period_;
        }
        last_raw_ = raw_abs;
        value_ += delta;
    }

    float value() const { return value_; }

   private:
    float period_;
    bool started_ = false;
    float last_raw_ = 0.0f;
    float value_ = 0.0f;
};

constexpr float kPeriod = 6.283185307179586f;
constexpr float kStroke = 6.57f;  // i2rt's measured linear_4310 stroke

// Fold a true multi-turn position into what a single-turn encoder reports.
float single_turn(float truth) {
    float wrapped = std::fmod(truth, kPeriod);
    if (wrapped < 0) wrapped += kPeriod;
    return wrapped;
}

TEST(ServoTurnTracking, TracksAStrokeLongerThanOneFeedbackTurn) {
    TurnAccumulator accumulator(kPeriod);
    // Closed at 0, opening to 6.57 -- past the wrap at 6.283.
    for (int step = 0; step <= 100; step++) {
        accumulator.accumulate(single_turn(kStroke * step / 100.0f));
    }
    EXPECT_NEAR(accumulator.value(), kStroke, 1e-3);
}

TEST(ServoTurnTracking, TheOpenEndDoesNotAliasOntoTheClosedEnd) {
    // The failure this exists to prevent: without accumulation the open stop
    // reads as ~0.287, indistinguishable from a gripper that is nearly shut.
    EXPECT_NEAR(single_turn(kStroke), 0.287f, 1e-3);

    TurnAccumulator accumulator(kPeriod);
    for (int step = 0; step <= 200; step++) {
        accumulator.accumulate(single_turn(kStroke * step / 200.0f));
    }
    EXPECT_GT(accumulator.value(), 6.0f);
}

TEST(ServoTurnTracking, CountsBackDownThroughTheWrap) {
    TurnAccumulator accumulator(kPeriod);
    for (int step = 0; step <= 100; step++) {
        accumulator.accumulate(single_turn(kStroke * step / 100.0f));
    }
    for (int step = 100; step >= 0; step--) {
        accumulator.accumulate(single_turn(kStroke * step / 100.0f));
    }
    EXPECT_NEAR(accumulator.value(), 0.0f, 1e-3);
}

TEST(ServoTurnTracking, SeedsOnTheFirstSampleRatherThanAtZero) {
    // The frame's origin is wherever the servo was on the first sample, which
    // is exactly why the startup calibration measures both stops in it.
    TurnAccumulator accumulator(kPeriod);
    accumulator.accumulate(4.2f);
    EXPECT_NEAR(accumulator.value(), 4.2f, 1e-6);
}

TEST(ServoTurnTracking, DisabledPeriodIsAPlainAssignment) {
    TurnAccumulator accumulator(0.0f);
    accumulator.accumulate(1.5f);
    accumulator.accumulate(6.2f);
    EXPECT_NEAR(accumulator.value(), 6.2f, 1e-6);
}

}  // namespace
