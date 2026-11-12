"""Hardware readiness checks must not accidentally power up an arm."""

import json
from types import SimpleNamespace

import pytest

from openpi_control import cli
from openpi_control.exceptions import ConfigurationError
from openpi_control.rigs import resolve_rig


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("KARMA_LOG_DIR", str(tmp_path / "logs"))


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("ERROR-ACTIVE", "ok"),
        ("BUS-OFF", "fail"),
        ("STOPPED", "fail"),
        ("SLEEPING", "fail"),
        ("ERROR-PASSIVE", "fail"),
        ("ERROR-WARNING", "fail"),
        ("FUTURE-STATE", "warn"),
    ],
)
def test_controller_state_matters_even_when_link_is_up(monkeypatch, state, status):
    payload = [{"linkinfo": {"info_data": {"state": state}}}]
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
        ),
    )
    monkeypatch.setattr(cli.buses, "check_interface", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_can_sysfs", lambda *a: "up")
    monkeypatch.setattr(cli, "can_bitrate", lambda *a: 1000000)
    checks = cli.run_doctor("Yam", "can0", effector_model="E_Yam")
    assert next(c for c in checks if c.name == "CAN controller").status == status


@pytest.mark.parametrize("payload", ["not json", "[]", "{}", "null", '[{"linkinfo": null}]'])
def test_unavailable_netlink_status_is_not_a_pass(monkeypatch, payload):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=payload,
        ),
    )
    assert cli.check_can_controller("can0").status == "warn"


def test_missing_ip_is_reported(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("ip")

    monkeypatch.setattr(cli.subprocess, "run", missing)
    assert cli.check_can_controller("can0").status == "warn"


def test_duplicate_bus_fails_before_any_backend_is_constructed():
    rig = resolve_rig("yam_bimanual").with_interfaces({"right": "can0"})

    def forbidden(*a, **k):
        pytest.fail("must not construct a backend on a duplicated bus")

    failures, _ = cli.preflight_rig(rig)
    assert failures == 1
    with pytest.raises(ConfigurationError, match="share can0"):
        cli.power_up(rig, backend_factory=forbidden)
    # Arm selection comes first: right alone on can0 remains a valid setup.
    assert cli.check_rig_interfaces(rig.subset(("right",))).status == "ok"


def test_serial_aliases_cannot_open_one_bus_twice(tmp_path):
    device = tmp_path / "ttyACM0"
    device.touch()
    alias = tmp_path / "by-id"
    alias.symlink_to(device)
    rig = resolve_rig("so101_bimanual").with_interfaces(
        {
            "left": str(device),
            "right": str(alias),
        }
    )
    assert cli.check_rig_interfaces(rig).status == "fail"


def test_report_preserves_failures_and_never_opens_a_bus(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("static report must not open hardware")

    monkeypatch.setattr(cli.buses, "open_bus", forbidden)
    report_path = tmp_path / "nested" / "doctor.json"
    assert (
        cli.main(
            [
                "doctor",
                "--model",
                "Yam",
                "--effector",
                "E_Yam",
                "--interface",
                "can_missing",
                "--report",
                str(report_path),
            ]
        )
        == 1
    )
    report = json.loads(report_path.read_text())
    assert report["schema"] == "karma.doctor.v1"
    assert report["failures"] >= 1
    assert report["physical_validation"] == "not_performed"
    assert report["probe_requested"] == "none"
    assert report["groups"][0]["interface"] == "can_missing"


def test_duplicate_rig_is_reported_without_probing(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("duplicate ownership must suppress probing")

    monkeypatch.setattr(cli.buses, "open_bus", forbidden)
    monkeypatch.setattr(cli, "run_camera_checks", lambda *a, **k: [])
    report_path = tmp_path / "rig.json"
    assert (
        cli.main(
            [
                "doctor",
                "--rig",
                "yam_bimanual",
                "--interface-override",
                "right=can0",
                "--probe",
                "--report",
                str(report_path),
            ]
        )
        == 1
    )
    report = json.loads(report_path.read_text())
    assert report["groups"][0]["checks"][0]["status"] == "fail"
    assert {g["name"] for g in report["groups"]} == {"rig", "left", "right", "cameras"}


def test_report_write_failure_has_operator_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_doctor", lambda *a, **k: [])
    assert (
        cli.main(
            [
                "doctor",
                "--model",
                "Yam",
                "--interface",
                "can0",
                "--report",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "cannot write doctor report" in capsys.readouterr().err


def test_warning_only_report_has_success_exit_but_no_physical_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda *a, **k: [
            cli.CheckResult("ok", "link state", "up"),
            cli.CheckResult("warn", "bus probe", "no traffic; idle DM bus"),
        ],
    )
    report_path = tmp_path / "warnings.json"
    assert (
        cli.main(
            [
                "doctor",
                "--model",
                "Yam",
                "--interface",
                "can0",
                "--probe",
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report["failures"] == 0
    assert report["warnings"] == 1
    assert report["probe_requested"] == "passive-listen"
    assert report["physical_validation"] == "not_performed"
