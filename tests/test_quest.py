"""Tests for the workstation-side Quest transport helpers."""

from __future__ import annotations

from subprocess import CompletedProcess
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_control import cli, quest
from openpi_control.exceptions import ConfigurationError


def adb_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> CompletedProcess[str]:
    return CompletedProcess(["adb"], returncode, stdout, stderr)


def test_adb_tunnel_selects_the_only_authorized_device(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(list(args))
        if args == ["devices"]:
            return adb_result("List of devices attached\nquest-serial\tdevice\n")
        return adb_result()

    monkeypatch.setattr(quest, "_run_adb", fake_run)
    tunnel = quest.QuestAdbTunnel(port=8443)

    assert tunnel.connect() == "quest-serial"
    tunnel.open_page("http://localhost:8443/")
    tunnel.close()

    assert calls == [
        ["devices"],
        ["-s", "quest-serial", "reverse", "tcp:8443", "tcp:8443"],
        [
            "-s",
            "quest-serial",
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            "http://localhost:8443/",
        ],
        ["-s", "quest-serial", "reverse", "--remove", "tcp:8443"],
    ]


def test_adb_tunnel_requires_a_serial_when_multiple_devices_are_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        quest,
        "_run_adb",
        lambda args: adb_result("List of devices attached\nquest-a\tdevice\nquest-b\tdevice\n"),
    )

    with pytest.raises(ConfigurationError, match="--adb-serial"):
        quest.QuestAdbTunnel(port=8443).connect()


def test_adb_tunnel_reports_unauthorized_headsets(monkeypatch) -> None:
    monkeypatch.setattr(
        quest,
        "_run_adb",
        lambda args: adb_result("List of devices attached\nquest-a\tunauthorized\n"),
    )

    with pytest.raises(ConfigurationError, match="accept the USB debugging prompt"):
        quest.QuestAdbTunnel(port=8443).connect()


def test_openpi_teleop_parses_single_arm_transport_and_ik_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))
    seen = {}

    def fake_command(args, log_path):
        seen["args"] = args
        seen["log_path"] = log_path
        return 0

    monkeypatch.setattr(cli, "_command_teleop", fake_command)

    assert (
        cli.main(
            [
                "teleop",
                "--arm",
                "right",
                "--interface",
                "right=can7",
                "--quest-transport",
                "lan",
                "--no-relay",
                "--vr-url",
                "wss://robot.example/ws",
                "--lam",
                "0.1",
                "--max-dq-pos",
                "0.02",
                "--rest-pose-right",
                "1,2,3,4,5,6,0",
                "--no-force-haptics",
            ]
        )
        == 0
    )

    args = seen["args"]
    assert args.arm == "right"
    assert args.interface == ["right=can7"]
    assert args.quest_transport == "lan"
    assert args.no_relay
    assert args.vr_url == "wss://robot.example/ws"
    assert cli._teleop_config(args) == {
        "id": "openpi-teleop",
        "lam": 0.1,
        "force_haptic_enabled": False,
        "rest_qpos_right": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "max_dq_per_joint_scalar_pos": 0.02,
        "max_dq_per_joint_scalar_rot": 0.24,
        "max_dq_per_joint": [0.02, 0.02, 0.02, 0.24, 0.24, 0.24],
    }


def test_collect_selects_one_arm_and_its_matching_camera_set(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENPI_LOG_DIR", str(tmp_path / "logs"))
    seen = {}

    def fake_collect(rig, **kwargs):
        seen["rig"] = rig
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_collect", fake_collect)

    assert (
        cli.main(
            [
                "collect",
                "--dry-run",
                "--arm",
                "right",
                "--interface",
                "right=can_right",
                "--format",
                "lerobot-v3",
                "--quest-transport",
                "lan",
                "--no-relay",
                "--no-viz",
                "--skip-preflight",
            ]
        )
        == 0
    )

    rig = seen["rig"]
    assert rig.names == ("right",)
    assert rig.camera_names == ("top", "right_wrist")
    assert all(camera.pixel_format == "rgb8" for camera in rig.cameras)
    assert seen["dataset_format"] == "lerobot-v3"
    assert seen["visualize"] is False


def test_collect_shares_one_reader_set_with_relay_recorder_and_teardown(
    monkeypatch,
) -> None:
    from openpi_control import cameras as cameras_mod
    from openpi_control.rigs import resolve_rig

    rig = resolve_rig("yam_bimanual")
    readers = {
        name: SimpleNamespace(spec=SimpleNamespace(name=name), stop=lambda: None)
        for name in rig.camera_names
    }
    events = []
    seen = {}

    monkeypatch.setattr(
        cameras_mod,
        "discover",
        lambda cameras, overrides=None: SimpleNamespace(
            complete=True,
            missing={},
            specs=lambda: tuple(camera for camera in cameras),
        ),
    )
    monkeypatch.setattr(cameras_mod, "open_readers", lambda specs: readers)
    monkeypatch.setattr(
        cameras_mod,
        "close_readers",
        lambda value: events.append(("cameras-stop", value)),
    )

    class FakeRelay:
        page_url = "http://127.0.0.1:8443/"

        def __init__(self, **kwargs):
            seen["relay_readers"] = kwargs["camera_readers"]

        def start(self):
            events.append("relay-start")

        def stop(self):
            events.append("relay-stop")

    class FakeAdb:
        def __init__(self, port, serial=None):
            del port, serial

        def connect(self):
            events.append("adb-start")
            return "quest"

        def open_page(self, url):
            events.append(("quest-page", url))

        def close(self):
            events.append("adb-stop")

    monkeypatch.setattr(quest, "QuestRelay", FakeRelay)
    monkeypatch.setattr(quest, "QuestAdbTunnel", FakeAdb)

    def fake_record(record_rig, **kwargs):
        seen["record_rig"] = record_rig
        seen["record_readers"] = kwargs["camera_readers"]
        events.append("record")
        return 0

    monkeypatch.setattr(cli, "run_record", fake_record)

    assert (
        cli.run_collect(
            rig,
            task="fold",
            repo_id=None,
            dry_run=True,
            open_quest=True,
        )
        == 0
    )

    assert seen["relay_readers"] is readers
    assert seen["record_readers"] is readers
    assert events[:4] == [
        "relay-start",
        "adb-start",
        ("quest-page", "http://127.0.0.1:8443/"),
        "record",
    ]
    assert events[-3:] == ["adb-stop", "relay-stop", ("cameras-stop", readers)]


def test_relay_accepts_rgb_collection_readers_without_owning_them() -> None:
    from vr_teleop_kit.relay import server

    class Reader:
        pixel_format = "rgb8"
        spec = SimpleNamespace(
            name="top",
            label="Top-down",
            width=640,
            height=480,
            rotate=0,
        )

        def latest(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def stop(self):
            raise AssertionError("the relay must not close a borrowed reader")

    reader = Reader()
    try:
        server.configure_camera_readers({"top": reader})
        track = server.CameraTrack(reader)

        assert server.CAMERA_READERS == {"top": reader}
        assert server._camera_id(server.CAMERA_SPECS[0]) == "top"
        assert track._av_format == "rgb24"
    finally:
        server.configure_camera_readers(None)
