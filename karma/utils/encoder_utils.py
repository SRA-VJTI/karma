"""Utility functions for working with i2rt encoders.

This module provides wrapper functions to work around compatibility issues
between karma and i2rt encoder implementations.
"""

from i2rt.motor_drivers.dm_driver import EncoderChain, PassiveEncoderReader, ReceiveMode


def get_encoder_chain_for_teaching_handle(can_interface):
    """Create an encoder chain for reading YAM teaching handle.

    This is a wrapper around i2rt's get_encoder_chain that works around
    the missing .channel attribute on CanInterface by manually creating
    the PassiveEncoderReader without validation.

    Args:
        can_interface: A CanInterface object (from DMChainCanInterface)

    Returns:
        EncoderChain configured for teaching handle (encoder ID 0x50E)
    """
    # Create PassiveEncoderReader without calling __init__ to skip validation
    # which requires .channel attribute that CanInterface doesn't have
    passive_reader = object.__new__(PassiveEncoderReader)
    passive_reader.can_interface = can_interface
    passive_reader.receive_mode = ReceiveMode.plus_one
    passive_reader.range_rad = 0.7

    # Create EncoderChain with encoder ID 0x50E (teaching handle encoder)
    encoder_chain = EncoderChain([0x50E], passive_reader)
    return encoder_chain
