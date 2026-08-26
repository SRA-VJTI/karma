"""Teach a raw-only MolmoAct server's ``_to_pil`` to decode encoded frames.

Twelve lines, added in place. The alternative -- copying this tree's
``examples/yam/host_server_yam.py`` over the deployment -- regresses whatever
that box's fork is *newer* at (``--revision`` pinning, request ``seed``, load
and GPU telemetry) to fix twelve lines, so it is the wrong trade.

Runs on the server box with nothing but the standard library, because that box
has the model environment and not necessarily this package:

    scp scripts/patch_server_to_pil.py USER@HOST:/tmp/
    ssh USER@HOST 'python3 /tmp/patch_server_to_pil.py \\
        ~/molmoact2/examples/yam/host_server_yam.py'

Idempotent: a server that already decodes encoded frames is left alone. The
original is kept beside the file as ``.bak`` and the result is syntax-checked
before it is written, so a failed match changes nothing. **Restart the server
afterwards** -- the running process is holding the old code.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

# The clause that tells the two generations apart, in the raised message. The
# client keys its raw-frames fallback on the *absence* of this text, so a patch
# that decodes encoded frames without advertising them would still be treated
# as raw-only by every client.
MARKER = "1-D uint8 (encoded)"

# The raw-only guard, whatever it is named and however it is spaced.
GUARD = re.compile(
    r"^(?P<indent>[ \t]+)if\s+(?P<var>\w+)\.ndim\s*!=\s*3.*?:\n"
    r"(?P<body>(?:[ \t]+.*\n)+?)"
    r"(?=^(?P=indent)\S|\Z)",
    re.MULTILINE,
)

BRANCH = """{indent}# Compressed transport: a 1-D uint8 array is an encoded image (JPEG/PNG)
{indent}# rather than a raw frame. json_numpy base64s ndarrays natively, so the
{indent}# client can send `np.frombuffer(jpeg_bytes, np.uint8)` with no new payload
{indent}# keys and no wire-format change. Raw HxWx3 frames still work unchanged, so
{indent}# an older client talking to this server is unaffected.
{indent}# Motivation: 3 raw 360x640 frames are ~2.8 MB of base64 per request, which
{indent}# dominates latency over Wi-Fi; JPEG cuts that by ~25x.
{indent}if {var}.ndim == 1:
{indent}    if {var}.dtype != np.uint8:
{indent}        {var} = {var}.astype(np.uint8)
{indent}    return Image.open(io.BytesIO({var}.tobytes())).convert("RGB")
"""


def patch_source(source: str) -> tuple[str, str]:
    """Return the patched source and a one-line description of what changed."""
    if MARKER in source:
        return source, "already decodes encoded frames"

    start = source.find("def _to_pil")
    if start < 0:
        raise SystemExit("no _to_pil in this file -- is it host_server_yam.py?")
    match = GUARD.search(source, start)
    if match is None:
        raise SystemExit(
            "could not find the `if <arr>.ndim != 3` guard inside _to_pil; "
            "patch it by hand against examples/yam/host_server_yam.py"
        )

    indent, var = match.group("indent"), match.group("var")
    patched = (
        source[: match.start()] + BRANCH.format(indent=indent, var=var) + source[match.start() :]
    )

    # Advertise the new form in the message the client reads. Only the guard's
    # own raise is touched, and only the shape clause inside it.
    def widen(m: re.Match[str]) -> str:
        return m.group(0).replace(
            "image must be HxWx3", f"image must be HxWx3 (raw) or {MARKER}", 1
        )

    patched, count = re.subn(r'raise ValueError\((?:f?".*?"|f?\'.*?\')\)', widen, patched, count=1)
    if count != 1 or MARKER not in patched:
        raise SystemExit(
            "added the decode branch but could not rewrite the ValueError message; "
            "nothing written -- patch by hand so the message keeps the "
            f"'{MARKER}' clause clients key their fallback on"
        )

    # io is used by the new branch; every generation of this file imports numpy
    # and PIL already, so io is the only one that can be missing. It goes after
    # `from __future__`, which the language requires to come first -- inserting
    # ahead of the first import unconditionally produces a file that will not
    # parse, which is what the compile() check below exists to catch.
    if not re.search(r"^import io$", patched, re.MULTILINE):
        future = re.search(r"^from __future__ import .*\n", patched, re.MULTILINE)
        at = future.end() if future else _first_import(patched)
        patched = patched[:at] + "import io\n" + patched[at:]
        return patched, "added the decode branch, widened the message, added `import io`"
    return patched, "added the decode branch and widened the message"


def _first_import(source: str) -> int:
    match = re.search(r"^(?:import|from) ", source, re.MULTILINE)
    if match is None:
        raise SystemExit("no imports in this file -- is it host_server_yam.py?")
    return match.start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, help="path to host_server_yam.py on this box")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would change and write nothing"
    )
    args = parser.parse_args()

    source = args.path.read_text()
    patched, what = patch_source(source)
    if patched == source:
        print(f"{args.path}: {what} — nothing to do")
        return 0

    try:
        compile(patched, str(args.path), "exec")
    except SyntaxError as err:
        raise SystemExit(f"patched source does not parse ({err}); nothing written") from err

    if args.dry_run:
        print(f"{args.path}: would have {what} (dry run, nothing written)")
        return 0

    backup = args.path.with_suffix(args.path.suffix + ".bak")
    shutil.copy2(args.path, backup)
    args.path.write_text(patched)
    print(f"{args.path}: {what}")
    print(f"  original kept at {backup}")
    print("  RESTART THE SERVER -- the running process still holds the old code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
