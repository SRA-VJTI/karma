/*!
 * @file pi_device_arm_can.cpp
 * @brief Implementation of the DeviceArmCan class for MIT-mode CAN arm device control.
 */

#include <unistd.h>

#include "pi_device_arm_can.hpp"

DeviceArmCan::DeviceArmCan(const CommandLineArgs& cla) : DeviceArm(cla) {}

DeviceArmCan::~DeviceArmCan() {}

ReturnCode DeviceArmCan::set_control_mode(Role target_role, ControlModeIntent intent) {
    // The DM servo mode methods are no-ops, but the device-level state reset
    // (including teaching cancellation on recovery) must still run.
    return DeviceArm::set_control_mode(target_role, intent);
}
