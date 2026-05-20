"""Karma command-line interface."""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
from pathlib import Path
from typing import Sequence


def _force_exit(sig, frame) -> None:  # noqa: ANN001
    """SIGTERM handler: give session.stop() 3 s then hard-kill the process group."""
    import threading
    import time

    def _kill_group() -> None:
        time.sleep(3.0)
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except Exception:
            os._exit(1)

    threading.Thread(target=_kill_group, daemon=True).start()


signal.signal(signal.SIGTERM, _force_exit)


def _run_session(args: argparse.Namespace) -> None:
    session_arg: str = args.config

    is_yaml = session_arg.endswith(('.yaml', '.yml'))
    is_file = os.path.exists(session_arg)

    if is_yaml or is_file:
        from karma.runtime.config import load_session

        try:
            session = load_session(session_arg)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Error loading session config '{session_arg}': {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Advanced escape hatch: dotted Python module exporting make_session().
        try:
            mod = importlib.import_module(session_arg)
        except ModuleNotFoundError as exc:
            print(f"Error: could not import '{session_arg}': {exc}", file=sys.stderr)
            sys.exit(1)

        if not hasattr(mod, "make_session"):
            print(f"Error: '{session_arg}' has no make_session() function.", file=sys.stderr)
            sys.exit(1)
        session = mod.make_session()

    if args.save_root:
        session._save_root = Path(args.save_root)

    session.start()

    if args.no_tui:
        print(f"Session running. Ctrl-C to stop. Recordings → {session.save_root}")
        session.wait()
    else:
        from karma.runtime.tui import run_tui

        run_tui(session)

    os._exit(0)


def _run_replay(args: argparse.Namespace) -> None:
    from karma.replay import main as replay_main

    replay_argv = [args.episode_dir, "--port", str(args.port), "--speed", str(args.speed)]
    if args.scene is not None:
        replay_argv.extend(["--scene", args.scene])
    if args.task is not None:
        replay_argv.extend(["--task", args.task])
    replay_main(replay_argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krm",
        description="Karma: realtime control and policy runtime for YAM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    session_parser = subparsers.add_parser("session", help="Launch a YAML session config.")
    session_parser.add_argument("config", help="Path to a YAML session config, e.g. configs/bimanual_yam_leader.yaml")
    session_parser.add_argument("--save-root", default=None, help="Override the session save_root for recordings.")
    session_parser.add_argument("--no-tui", action="store_true", help="Disable the Rich TUI and block until Ctrl-C.")
    session_parser.set_defaults(func=_run_session)

    replay_parser = subparsers.add_parser("replay", help="Replay a recorded sim episode in Viser.")
    replay_parser.add_argument("episode_dir", help="Path to episode directory")
    replay_parser.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080)")
    replay_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    replay_parser.add_argument("--scene", type=str, default=None, help="Scene name (default: from session_meta or 'hybrid')")
    replay_parser.add_argument("--task", type=str, default=None, help="Task name (default: from session_meta or 'bottles')")
    replay_parser.set_defaults(func=_run_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
