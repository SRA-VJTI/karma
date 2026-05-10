"""YAML session loader — maps node type strings to Node classes.

Session YAML schema:
    version: "1"
    session:
      save_root: recordings
      auto_record_duration: 10.0   # optional
      record_topic: null            # optional, e.g. "leader_left/record"
    nodes:
      - type: DummyGelloNode
        name: leader_left
        arm: left
        ...
      - type: XdofSimNode
        name: sim
        cmd_topics:
          left: leader_left/joint_pos
          right: leader_right/joint_pos
        ...

Alternatively, pass a Python module dotted path (no .yaml extension) that
exports a make_session() function — for backward compatibility.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karma.runtime.session import Session


# ── Node registry ─────────────────────────────────────────────────────────────

_NODE_REGISTRY: dict[str, str] = {
    "AgentNode":        "karma.runtime.agent_node:AgentNode",
    "RobotNode":        "karma.runtime.environment.robot_node:RobotNode",
    "CameraNode":       "karma.runtime.environment.camera_node:CameraNode",
    "XdofSimNode":      "karma.runtime.sim.xdof_sim_node:XdofSimNode",
    "ViserTeleopNode":  "karma.runtime.viser_teleop_node:ViserTeleopNode",
    "ViserMonitorNode": "karma.runtime.viser_monitor_node:ViserMonitorNode",
}


def _resolve_node_cls(type_name: str):
    """Resolve a node class from the registry or a dotted module path."""
    if type_name in _NODE_REGISTRY:
        ref = _NODE_REGISTRY[type_name]
    elif ":" in type_name:
        ref = type_name
    else:
        raise ValueError(
            f"Unknown node type '{type_name}'. "
            f"Known types: {list(_NODE_REGISTRY.keys())}"
        )
    module_path, cls_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


# ── Writer factory ────────────────────────────────────────────────────────────

def _make_writer_for_node(node_cls, node_params: dict):
    """Return an appropriate Writer for the given node type."""
    from karma.runtime.recording import McapWriter, AsyncMp4Writer, NullWriter
    from karma.runtime.node import NodeRole

    # XdofSimNode manages its own writers — give it NullWriter
    type_name = node_params.get("type", "")
    if type_name == "XdofSimNode":
        return NullWriter()

    # Check role
    role = getattr(node_cls, "role", NodeRole.ROBOT)
    if role == NodeRole.SENSOR:
        fps = float(node_params.get("fps", 30.0))
        return AsyncMp4Writer(fps=fps)

    return McapWriter()


# ── YAML loader ───────────────────────────────────────────────────────────────

def load_session(path: str) -> "Session":
    """Load a Session from a YAML config file or Python module path.

    Args:
        path: Path to a .yaml file, or a dotted Python module path
              containing make_session() (legacy).
    """
    # Python module path (legacy / advanced)
    if not path.endswith(".yaml") and not path.endswith(".yml") and os.path.sep not in path:
        # Try as a Python module first
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "make_session"):
                return mod.make_session()
        except (ImportError, ModuleNotFoundError):
            pass

    # YAML file
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Session config not found: {yaml_path}")

    return _load_from_yaml(yaml_path)


def _load_from_yaml(yaml_path: Path) -> "Session":
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required to load YAML session configs. "
                          "Install it with: pip install pyyaml") from e

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("version") != "1":
        raise ValueError(f"Unsupported session config version: {cfg.get('version')}")

    session_cfg: dict = cfg.get("session", {})
    save_root: str = session_cfg.get("save_root", "recordings")
    auto_record_duration: float | None = session_cfg.get("auto_record_duration")
    record_topic: str | None = session_cfg.get("record_topic")
    start_paused: bool = bool(session_cfg.get("start_paused", False))
    record_on_unpause: bool = bool(session_cfg.get("record_on_unpause", False))

    nodes_cfg: list[dict] = cfg.get("nodes", [])
    nodes = []

    for node_params in nodes_cfg:
        node_params = dict(node_params)  # copy to avoid mutation
        type_name: str = node_params.pop("type")
        node_cls = _resolve_node_cls(type_name)

        # Build constructor kwargs via classmethod
        kwargs = node_cls.build_kwargs({**node_params, "type": type_name})

        # Inject writer
        node_params_with_type = {**node_params, "type": type_name}
        writer = _make_writer_for_node(node_cls, node_params_with_type)
        kwargs["writer"] = writer

        try:
            node = node_cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate {type_name} with kwargs {kwargs}: {e}"
            ) from e

        nodes.append(node)

    from karma.runtime.session import Session

    return Session(
        nodes=nodes,
        save_root=save_root,
        record_topic=record_topic,
        auto_record_duration=auto_record_duration,
        start_paused=start_paused,
        record_on_unpause=record_on_unpause,
    )
