"""Quest transport helpers used by the :mod:`openpi_control` CLI.

The VR package owns the WebXR protocol and the relay application. This module
owns the workstation-side lifecycle around it: starting the relay in-process,
creating the USB ``adb reverse`` tunnel, and optionally opening the Quest
browser at the relay page. Keeping those actions here gives ``openpi teleop``
one predictable command while leaving ``openpi relay`` available for a relay
that should outlive a robot session.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .exceptions import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence


def _run_adb(args: Sequence[str], *, timeout_s: float = 10.0) -> subprocess.CompletedProcess[str]:
    """Run one non-interactive ADB command with an actionable error."""
    try:
        return subprocess.run(
            ["adb", *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as err:
        raise ConfigurationError(
            "ADB is not installed or is not on PATH; install Android platform-tools "
            "or use `--quest-transport lan`"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise ConfigurationError(f"ADB command timed out after {timeout_s:g}s") from err
    except OSError as err:
        raise ConfigurationError(f"could not run ADB: {err}") from err


def _adb_error(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout).strip()
    return detail or f"ADB exited with status {result.returncode}"


@dataclass(slots=True)
class QuestAdbTunnel:
    """An ADB reverse tunnel from a Quest headset to the relay port."""

    port: int
    serial: str | None = None
    _connected_serial: str | None = None

    def connect(self) -> str:
        result = _run_adb(["devices"])
        if result.returncode != 0:
            raise ConfigurationError(f"`adb devices` failed: {_adb_error(result)}")

        devices: dict[str, str] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices attached"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                devices[fields[0]] = fields[1]

        if self.serial is not None:
            state = devices.get(self.serial)
            if state != "device":
                state_text = state or "not found"
                raise ConfigurationError(
                    f"Quest ADB device {self.serial!r} is {state_text}; accept the USB "
                    "debugging prompt in the headset and retry"
                )
            selected = self.serial
        else:
            ready = [serial for serial, state in devices.items() if state == "device"]
            if len(ready) != 1:
                if not ready and devices:
                    states = ", ".join(f"{serial} ({state})" for serial, state in devices.items())
                    raise ConfigurationError(
                        f"no authorized Quest ADB device is ready ({states}); accept the "
                        "USB debugging prompt in the headset"
                    )
                if not ready:
                    raise ConfigurationError(
                        "no Quest ADB device found; connect the headset, enable Developer "
                        "Mode, and accept USB debugging"
                    )
                raise ConfigurationError(
                    "more than one ADB device is connected; select one with --adb-serial "
                    + ", ".join(ready)
                )
            selected = ready[0]

        result = _run_adb(["-s", selected, "reverse", f"tcp:{self.port}", f"tcp:{self.port}"])
        if result.returncode != 0:
            raise ConfigurationError(
                f"could not create Quest USB tunnel on tcp:{self.port}: {_adb_error(result)}"
            )
        self._connected_serial = selected
        return selected

    def open_page(self, url: str) -> None:
        """Ask the Quest's default browser to open ``url``."""
        if self._connected_serial is None:
            raise ConfigurationError("ADB tunnel is not connected")
        result = _run_adb(
            [
                "-s",
                self._connected_serial,
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
            ]
        )
        if result.returncode != 0:
            raise ConfigurationError(f"could not open Quest browser: {_adb_error(result)}")

    def close(self) -> None:
        """Remove the reverse rule, without masking a robot shutdown error."""
        if self._connected_serial is None:
            return
        try:
            result = _run_adb(
                ["-s", self._connected_serial, "reverse", "--remove", f"tcp:{self.port}"]
            )
            if result.returncode != 0:
                # A headset unplug during teardown is expected. There is no
                # useful recovery action, and the next connect will recreate it.
                return
        except ConfigurationError:
            return
        finally:
            self._connected_serial = None


@dataclass(slots=True)
class QuestRelay:
    """A background uvicorn server for the vendored Quest WebXR relay."""

    host: str = "127.0.0.1"
    port: int = 8443
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    camera_readers: Mapping[str, object] | None = None
    _server: object | None = None
    _thread: threading.Thread | None = None
    _error: BaseException | None = None

    @property
    def tls_enabled(self) -> bool:
        return self.ssl_keyfile is not None or self.ssl_certfile is not None

    @property
    def page_url(self) -> str:
        scheme = "https" if self.tls_enabled else "http"
        return f"{scheme}://127.0.0.1:{self.port}/"

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.tls_enabled and not (self.ssl_keyfile and self.ssl_certfile):
            raise ConfigurationError("Quest relay TLS needs both --ssl-keyfile and --ssl-certfile")

        try:
            import uvicorn

            from vr_teleop_kit.relay.server import app, configure_camera_readers
        except ImportError as err:
            raise ConfigurationError(
                "the Quest relay needs its optional dependencies; run `uv sync` "
                "to install the VR extra"
            ) from err

        # A collector already owns its RealSense streams. Lending those exact
        # readers to the in-process relay lets Quest WebRTC, Viser, and the
        # LeRobot writer consume one capture without fighting over the device.
        configure_camera_readers(self.camera_readers)

        self._error = None
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
        )
        server = uvicorn.Server(config)
        self._server = server

        def serve() -> None:
            try:
                server.run()
            except BaseException as err:  # propagate startup failures to start()
                self._error = err

        self._thread = threading.Thread(target=serve, name="openpi-quest-relay", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not getattr(server, "started", False):
            if self._error is not None:
                self._thread = None
                raise ConfigurationError(
                    f"Quest relay failed to start: {self._error}"
                ) from self._error
            if not self._thread.is_alive():
                self._thread = None
                raise ConfigurationError(
                    f"Quest relay exited before listening on {self.host}:{self.port}"
                )
            if time.monotonic() >= deadline:
                self.stop()
                raise ConfigurationError(
                    f"Quest relay did not start within 10s on {self.host}:{self.port}"
                )
            time.sleep(0.05)

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5.0)
        self._server = None
        self._thread = None


def run_relay_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8443,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
) -> int:
    """Run the vendored relay in the foreground for ``openpi relay``."""
    relay = QuestRelay(host, port, ssl_keyfile, ssl_certfile)
    try:
        relay.start()
        print(f"Quest relay listening at {relay.page_url}")
        print("ctrl-c to stop the relay")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        relay.stop()
