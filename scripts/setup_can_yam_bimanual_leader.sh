#!/bin/bash
# Setup CAN interfaces for YAM bimanual YAM leader teleoperation
# This script brings up all 4 CAN interfaces required for the system:
# - can_tleader_l: Left leader arm (teaching handle)
# - can_tleader_r: Right leader arm (teaching handle)
# - can_follow_l: Left follower arm
# - can_follow_r: Right follower arm

set -e

BITRATE=1000000
INTERFACES=(can_tleader_l can_tleader_r can_follow_l can_follow_r)

echo "Setting up CAN interfaces for YAM bimanual leader..."

for iface in "${INTERFACES[@]}"; do
    echo "Configuring $iface..."

    # Check if interface exists
    if ! ip link show "$iface" &> /dev/null; then
        echo "  ERROR: Interface $iface not found!"
        echo "  Make sure the CANable device is plugged in and has a persistent ID set."
        exit 1
    fi

    # Bring down first (in case it's already up)
    sudo ip link set "$iface" down 2>/dev/null || true

    # Configure and bring up
    sudo ip link set "$iface" type can bitrate $BITRATE
    sudo ip link set "$iface" up

    # Verify it's up
    if ip link show "$iface" | grep -q "state UP"; then
        echo "  ✓ $iface is UP"
    else
        echo "  ✗ Failed to bring up $iface"
        exit 1
    fi
done

echo ""
echo "All CAN interfaces are ready!"
echo ""
echo "Interface status:"
ip -brief link show | grep can

echo ""
echo "You can now run:"
echo "  uv run krm session configs/bimanual_yam_leader.yaml"
