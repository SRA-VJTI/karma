"""Tests for the servo driver registry and the per-family zeroing routines."""

from __future__ import annotations

import pathlib

import can
import pytest

from openpi_control import servos
from openpi_control.servos import buses, dm_can, ft_serial

_ENCOS_BROADCAST_ID = 0x7FF


class _FakeBus:
    """Minimal python-can bus double: records sends, replays scripted responses."""

    def __init__(self, responses: list[can.Message | None]) -> None:
        self.sent: list[can.Message] = []
        self._responses = responses

    def send(self, message: can.Message) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> can.Message | None:
        if not self._responses:
            return None
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _no_settle_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dm_can, "_POST_ZERO_SETTLE_S", 0.0)
    monkeypatch.setattr(encos_can, "_POST_ZERO_SETTLE_S", 0.0)
    monkeypatch.setattr(encos_can, "_INTER_COMMAND_GAP_S", 0.0)
    monkeypatch.setattr(encos_can, "_RESPONSE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(encos_can, "_DISABLE_ACK_TIMEOUT_S", 0.01)


def test_registry_resolves_every_model_json_servo() -> None:
    assert servos.zero_driver("DM J4340") is dm_can
    assert servos.zero_driver("FeeTech STS3215") is ft_serial
    assert servos.zero_driver("CAN Passive Encoder") is None


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit, match="not in the servo registry"):
        servos.zero_driver("Unknown Servo 9000")


def test_every_driver_declares_a_known_port_type_and_zero_entrypoint() -> None:
    for driver in servos.SERVO_ZERO_DRIVERS.values():
        if driver is None:
            continue
        assert driver.PORT_TYPE in (
            buses.PORT_TYPE_CAN,
            buses.PORT_TYPE_SERIAL,
            buses.PORT_TYPE_ETHERNET,
        )
        if driver.WHOLE_ARM_ZERO:
            assert callable(driver.set_zero_whole_arm)
        else:
            assert callable(driver.set_zero)


def test_dm_zero_acknowledged() -> None:
    bus = _FakeBus([can.Message(arbitration_id=0x01)])
    assert dm_can.set_zero(bus, 0x01) is None
    assert len(bus.sent) == 1
    assert bus.sent[0].arbitration_id == 0x01
    assert bytes(bus.sent[0].data) == bytes([0xFF] * 7 + [0xFE])


def test_dm_zero_no_ack_is_an_error() -> None:
    bus = _FakeBus([])
    assert dm_can.set_zero(bus, 0x01) == "no acknowledgement"


def test_serial_bus_session_requires_a_baudrate() -> None:
    with pytest.raises(SystemExit, match="explicit positive baudrate"):
        with buses.open_bus(buses.PORT_TYPE_SERIAL, "/dev/ttyUSB_test"):
            pass


def test_serial_bus_session_opens_pyserial_with_catalog_baudrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: dict[str, object] = {}

    class _FakePortSession:
        def __init__(self, *, port: str, baudrate: int, timeout: float) -> None:
            opened.update(port=port, baudrate=baudrate, timeout=timeout)

        def __enter__(self) -> _FakePortSession:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    monkeypatch.setattr(buses.pyserial, "Serial", _FakePortSession)
    with buses.open_bus(buses.PORT_TYPE_SERIAL, "/dev/ttyUSB_test", baudrate=1000000) as bus:
        assert isinstance(bus, _FakePortSession)
    assert opened["port"] == "/dev/ttyUSB_test"
    assert opened["baudrate"] == 1000000
    assert opened["timeout"] == pytest.approx(0.1)


# --- FeeTech SMS/STS (ft_serial) ------------------------------------------


class _FakeSerial:
    """Minimal pyserial double: records writes, replays scripted response bytes."""

    def __init__(self, responses: list[bytes]) -> None:
        self.written: list[bytes] = []
        self.baudrate = 1000000
        self._responses = responses
        self._buffer = b""

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._buffer = b""

    def read(self, size: int) -> bytes:
        if not self._buffer and self._responses:
            self._buffer = self._responses.pop(0)
        result, self._buffer = self._buffer[:size], self._buffer[size:]
        return result


def _ft_status(servo_id: int, *, error: int = 0, params: bytes = b"") -> bytes:
    """A FeeTech status packet: FF FF ID LEN ERR [params] CHK."""
    packet = bytearray(b"\xff\xff") + bytes([servo_id, 2 + len(params), error]) + params
    packet.append(ft_serial.check_sum(packet))
    return bytes(packet)


def test_ft_build_packet_checksum_matches_protocol() -> None:
    # PING to servo 1: FF FF 01 02 01 FB (complement checksum over ID..params).
    packet = ft_serial.build_packet(1, ft_serial.INST_PING, b"")
    assert packet == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_ft_ping_and_scan_report_answering_ids() -> None:
    # Servo 2 stays silent (empty response); 1 and 3 answer valid status packets.
    bus = _FakeSerial([_ft_status(1), b"", _ft_status(3)])
    assert ft_serial.scan(bus, (1, 2, 3)) == [1, 3]


def test_ft_read_returns_register_bytes() -> None:
    bus = _FakeSerial([_ft_status(1, params=bytes([0x04]))])
    assert ft_serial.ft_read(bus, 1, ft_serial.ADDR_BAUD_RATE, 1) == bytes([0x04])
    # The read request names the address and length.
    assert bus.written[0][5] == ft_serial.ADDR_BAUD_RATE
    assert bus.written[0][6] == 1


def test_ft_read_rejects_corrupted_checksum() -> None:
    good = _ft_status(1, params=bytes([0x04]))
    corrupted = good[:-1] + bytes([good[-1] ^ 0xFF])
    bus = _FakeSerial([corrupted])
    assert ft_serial.ft_read(bus, 1, ft_serial.ADDR_BAUD_RATE, 1) is None


def test_ft_write_reports_servo_error_bits() -> None:
    bus = _FakeSerial([_ft_status(1, error=0x20)])
    assert ft_serial.ft_write(bus, 1, ft_serial.ADDR_TORQUE_ENABLE, 1, 1) is False


def test_ft_set_zero_wraps_calibrate_middle_in_eeprom_unlock_lock() -> None:
    bus = _FakeSerial([_ft_status(2), _ft_status(2), _ft_status(2)])
    assert ft_serial.set_zero(bus, 2) is None
    # Packet layout: FF FF ID LEN INST ADDR VALUE CHK.
    addresses = [packet[5] for packet in bus.written]
    values = [packet[6] for packet in bus.written]
    assert addresses == [
        ft_serial.ADDR_LOCK_FLAG,
        ft_serial.ADDR_TORQUE_ENABLE,
        ft_serial.ADDR_LOCK_FLAG,
    ]
    assert values == [0, ft_serial.TORQUE_ENABLE_CALIBRATE_MIDDLE, 1]


def test_ft_set_zero_reports_unlock_failure() -> None:
    bus = _FakeSerial([])  # the unlock write times out
    error = ft_serial.set_zero(bus, 2)
    assert error is not None and "unlock" in error


def test_ft_baudrate_register_values_are_reversible() -> None:
    mapping = ft_serial.BAUDRATE_REG_VALUES
    assert mapping[1000000] == 0
    assert len(set(mapping.values())) == len(mapping)


def test_ft_get_baudrate_reads_the_register() -> None:
    bus = _FakeSerial([_ft_status(1, params=bytes([ft_serial.BAUDRATE_REG_VALUES[115200]]))])
    assert ft_serial.get_baudrate(bus, 1) == 115200


def test_ft_set_baudrate_switches_the_host_side_too() -> None:
    bus = _FakeSerial([_ft_status(1), _ft_status(1), _ft_status(1), _ft_status(1)])
    assert ft_serial.set_baudrate(bus, 1, 500000) is True
    assert bus.baudrate == 500000


def test_ft_set_baudrate_restores_host_baud_when_servo_goes_silent() -> None:
    # unlock ack + write ack, then silence: no lock ack, no ping answer.
    bus = _FakeSerial([_ft_status(1), _ft_status(1)])
    assert ft_serial.set_baudrate(bus, 1, 500000) is False
    assert bus.baudrate == 1000000


def test_ft_set_baudrate_rejects_unsupported_rates() -> None:
    bus = _FakeSerial([])
    assert ft_serial.set_baudrate(bus, 1, 12345) is False
    assert bus.written == []


def test_check_interface_serial_uses_device_path(tmp_path: pathlib.Path) -> None:
    device = tmp_path / "ttyUSB_test"
    assert buses.check_interface(buses.PORT_TYPE_SERIAL, str(device)) is not None
    device.touch()
    assert buses.check_interface(buses.PORT_TYPE_SERIAL, str(device)) is None


def test_check_interface_can_reports_missing_interface() -> None:
    error = buses.check_interface(buses.PORT_TYPE_CAN, "can_does_not_exist")
    assert error is not None and "does not exist" in error
