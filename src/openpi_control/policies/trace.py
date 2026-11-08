"""Small per-session artifacts; credentials and image payloads are never logged."""

import importlib.metadata
import json
import platform
import threading
import time
import uuid
from pathlib import Path

from .. import log_paths


class PolicyTrace:
    def __init__(self, config):
        self.path = (
            Path(log_paths.runtime_log_dir())
            / "policies"
            / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        )
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "config.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n")
        versions = {"python": platform.python_version()}
        for package in ("karma", "numpy", "requests", "viser"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed"
        (self.path / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
        self.lock = threading.Lock()
        self.stream = (self.path / "events.jsonl").open("a")

    def event(self, kind, **fields):
        with self.lock:
            if not self.stream.closed:
                self.stream.write(
                    json.dumps(
                        dict(event=kind, monotonic=time.monotonic(), **fields),
                        allow_nan=False,
                        default=str,
                    )
                    + "\n"
                )
                self.stream.flush()

    def close(self):
        with self.lock:
            self.stream.close()
