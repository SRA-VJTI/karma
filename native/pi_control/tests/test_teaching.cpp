#include <gtest/gtest.h>
#include <limits>
#include "pi_device.hpp"
#include "pi_joint.hpp"
#include "pi_servo_dm.hpp"
#include "pi_teaching.hpp"

namespace {
class TeachingServo : public ServoDm {
 public:
    TeachingServo(Device* device, Joint* joint) : ServoDm(device, joint, nullptr) {
        dir_invert_ = 1;
        pos_min_rel_ = -2;
        pos_max_rel_ = 2;
    }
    float applied = 0;
    int sends = 0;
    prof_time_msec_t age = 0;
    ReturnCode apply_torque(float torque) override { applied = torque; ++sends; return ReturnCode::SUCCESS; }
    ReturnCode park_safely() override { return ReturnCode::SUCCESS; }
    prof_time_msec_t get_frame_age_ms() override { return age; }
};

class TeachingDevice : public Device {
 public:
    TeachingServo* servo;
    bool rebased = false;
    explicit TeachingDevice(const CommandLineArgs& cla) : Device(cla) {
        role_ = Role::FOLLOWER;
        auto joint = std::make_unique<Joint>(this, nullptr);
        auto motor = std::make_unique<TeachingServo>(this, joint.get());
        servo = motor.get();
        joint->get_servos().push_back(std::move(motor));
        joints_.push_back(std::move(joint));
    }
    bool supports_teaching() const override { return true; }
    void expire() { teaching_renewed_ -= std::chrono::seconds(1); }
    ReturnCode apply_action(const MsgJoints&) override { return ReturnCode::SUCCESS; }
    ReturnCode get_observation(MsgJoints&) override { return ReturnCode::SUCCESS; }
    ReturnCode process_follower_msg(const MsgJoints&) override { return ReturnCode::SUCCESS; }
    ReturnCode read_hardware_values() override { return ReturnCode::SUCCESS; }
    ReturnCode write_hardware_values() override { return ReturnCode::SUCCESS; }
    ReturnCode move_to_ready_position() override { return ReturnCode::SUCCESS; }
    ReturnCode operate_as_leader() override { return ReturnCode::SUCCESS; }
    ReturnCode operate_as_follower() override { return ReturnCode::SUCCESS; }
    ReturnCode get_servo_ids(std::vector<int>&) override { return ReturnCode::SUCCESS; }
    ReturnCode set_control_mode(Role, ControlModeIntent) override { return ReturnCode::SUCCESS; }
    void clear_command_buffers_for_move_to_ready() override { rebased = true; }
};

TeachingJoint profile() { return TeachingJoint{0.1f, 0.1f, 0.05f, 0.2f, 1.0f, 0.05f, true}; }

TEST(Teaching, RejectsInvalidAndUnconfiguredProfiles) {
    CommandLineArgs cla{};
    TeachingDevice device(cla);
    EXPECT_EQ(device.set_runtime_teaching(true), ReturnCode::INVALID_PARAM);
    auto p = profile(); p.transition = 0;
    EXPECT_EQ(device.configure_teaching_joint(0, p), ReturnCode::INVALID_PARAM);
    p = profile(); p.coulomb = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(device.configure_teaching_joint(0, p), ReturnCode::INVALID_PARAM);
    EXPECT_EQ(device.configure_teaching_joint(1, profile()), ReturnCode::INVALID_PARAM);
}

TEST(Teaching, SmoothOddBoundedAssistAndZeroTorqueAtRest) {
    const auto p = profile();
    EXPECT_FLOAT_EQ(p.torque(0), 0);
    EXPECT_NEAR(p.torque(0.1), -p.torque(-0.1), 1e-7);
    EXPECT_LT(std::fabs(p.torque(1e-6)), 1e-4);
    EXPECT_LE(std::fabs(p.torque(100) + p.damping * 100), p.compensation_limit + 1e-6);
}

TEST(Teaching, PreservesDampingSignForInvertedGripper) {
    CommandLineArgs cla{};
    for (int direction : {-1, 1}) {
        TeachingDevice device(cla);
        device.servo->dir_invert_ = direction;
        device.servo->curr_vel_ = 0.2f;
        TeachingJoint p{}; p.damping = 0.1f;
        ASSERT_EQ(device.configure_teaching_joint(0, p), ReturnCode::SUCCESS);
        ASSERT_EQ(device.set_runtime_teaching(true), ReturnCode::SUCCESS);
        ASSERT_EQ(device.apply_teaching_torques({0.3f}), ReturnCode::SUCCESS);
        EXPECT_NEAR(device.servo->applied, 0.28f, 1e-6);
    }
}

TEST(Teaching, LeaseExpirationHoldsAndCannotBeRenewedBackIntoTeaching) {
    CommandLineArgs cla{};
    TeachingDevice device(cla);
    ASSERT_EQ(device.configure_teaching_joint(0, profile()), ReturnCode::SUCCESS);
    ASSERT_EQ(device.set_runtime_teaching(true), ReturnCode::SUCCESS);
    device.expire();
    EXPECT_EQ(device.renew_teaching(), ReturnCode::SUCCESS);
    EXPECT_FALSE(device.teaching_active());
    EXPECT_TRUE(device.rebased);
    EXPECT_EQ(device.renew_teaching(), ReturnCode::BUSY);
    EXPECT_EQ(device.servo->sends, 0);
}

TEST(Teaching, OverspeedStaleAndOutOfRangeFeedbackStopBeforeTorque) {
    CommandLineArgs cla{};
    for (int fault = 0; fault < 5; ++fault) {
        TeachingDevice device(cla);
        ASSERT_EQ(device.configure_teaching_joint(0, profile()), ReturnCode::SUCCESS);
        ASSERT_EQ(device.set_runtime_teaching(true), ReturnCode::SUCCESS);
        if (fault == 0) device.servo->curr_vel_ = 2;
        if (fault == 1) device.servo->age = 101;
        if (fault == 2) device.servo->curr_pos_abs_ = 3;
        if (fault == 3) device.servo->age = -1;
        if (fault == 4) device.expire();
        EXPECT_EQ(device.apply_teaching_torques({0}), ReturnCode::SUCCESS);
        EXPECT_FALSE(device.teaching_active());
        EXPECT_EQ(device.servo->sends, 0);
    }
}

TEST(Teaching, CannotRetuneAnActiveDevice) {
    CommandLineArgs cla{};
    TeachingDevice device(cla);
    ASSERT_EQ(device.configure_teaching_joint(0, profile()), ReturnCode::SUCCESS);
    ASSERT_EQ(device.set_runtime_teaching(true), ReturnCode::SUCCESS);
    EXPECT_EQ(device.configure_teaching_joint(0, profile()), ReturnCode::BUSY);
    EXPECT_EQ(device.runtime_hold(), ReturnCode::SUCCESS);
    EXPECT_EQ(device.configure_teaching_joint(0, profile()), ReturnCode::SUCCESS);
}
}  // namespace
