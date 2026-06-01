#!/usr/bin/env python3
"""Forward karma's per-node session logs to a remote collector (e.g. the GPU box).

karma's Session writes one logfile per node into a fresh temp dir each run
(``/tmp/karma_logs_XXXX/<node>.log``). This script finds the newest such dir,
tails every ``*.log`` in it, prefixes each line with its node name, and streams
the lines over TCP to a collector so the whole session can be monitored from
another machine.

Run on the YAMbox alongside ``krm session``:

    python3 scripts/karma_log_forward.py --sink 192.168.0.209:9020

Pairs with rltoken/tools/karma_log_sink.py on the GPU box. Stdlib only; no karma
imports, so it runs under plain python3. Reconnects to the sink if it drops, and
picks up new node logfiles as they appear.
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import threading
import time
from typing import Dict, Optional


def _newest_log_dir(pattern: str, wait: bool) -> str:
    while True:
        dirs = sorted(glob.glob(pattern), key=lambda d: os.path.getmtime(d) if os.path.exists(d) else 0)
        if dirs:
            return dirs[-1]
        if not wait:
            raise SystemExit(f"no karma log dir matching {pattern!r}; start the session first or pass --glob")
        print(f"[forward] waiting for a log dir matching {pattern} ...")
        time.sleep(2.0)


class _Sink:
    """TCP line sender with lazy reconnect."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, int(port)
        self.sock: Optional[socket.socket] = None

    def send(self, line: str) -> None:
        data = (line.rstrip("\n") + "\n").encode("utf-8", "replace")
        for _ in range(2):
            try:
                if self.sock is None:
                    self.sock = socket.create_connection((self.host, self.port), timeout=5)
                    print(f"[forward] connected to sink {self.host}:{self.port}")
                self.sock.sendall(data)
                return
            except OSError:
                try:
                    if self.sock:
                        self.sock.close()
                except OSError:
                    pass
                self.sock = None
                time.sleep(1.0)


def _tail_file(path: str, node: str, sink: _Sink, from_start: bool) -> None:
    with open(path, "r", errors="replace") as fh:
        if not from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if line:
                sink.send(f"[{node}] {line.rstrip()}")
            else:
                time.sleep(0.2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sink", required=True, help="collector host:port, e.g. 192.168.0.209:9020")
    p.add_argument("--glob", default="/tmp/karma_logs_*", help="karma session log-dir glob")
    p.add_argument("--from-start", action="store_true", help="forward existing log contents too")
    args = p.parse_args()

    host, _, port = args.sink.partition(":")
    sink = _Sink(host, port or "9020")
    log_dir = _newest_log_dir(args.glob, wait=True)
    print(f"[forward] tailing {log_dir} -> {args.sink}")

    tailing: Dict[str, threading.Thread] = {}
    while True:
        for path in glob.glob(os.path.join(log_dir, "*.log")):
            if path not in tailing:
                node = os.path.splitext(os.path.basename(path))[0]
                t = threading.Thread(target=_tail_file, args=(path, node, sink, args.from_start), daemon=True)
                t.start()
                tailing[path] = t
                print(f"[forward] + tailing node '{node}'")
        time.sleep(2.0)


if __name__ == "__main__":
    main()
