"""Ask a MolmoAct server whether it can decode encoded frames, over HTTP only.

The client drops a whole run to raw frames the first time a server complains
about a 1-D frame shape (``EncodedFramesUnsupported``), which costs ~14x the
payload for the rest of the episode. That is a property of the server, not of
the run, so it is worth knowing before two arms are energized rather than from
a stderr line in the middle of a fold.

One inference each way -- encoded and raw -- against synthetic black frames.
No hardware is touched and nothing is commanded.

    uv run python scripts/probe_encoded_frames.py 192.168.0.107:4090
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from openpi_control.inference import (
    DEFAULT_MOLMOACT_JPEG_QUALITY,
    BimanualObservation,
    EncodedFramesUnsupported,
    InferenceError,
    MolmoActClient,
    normalize_server_url,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", help="server URL or host:port; /act is appended")
    parser.add_argument(
        "--instruction",
        default="probe",
        help="instruction sent with the probe (the actions are discarded)",
    )
    # 360x640 is what the reference deployment's frames are resized to; the
    # probe only needs a shape the server will accept, not the rig's own.
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = normalize_server_url(args.server)
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    observation = BimanualObservation(
        top_cam=frame, left_cam=frame, right_cam=frame, state=np.zeros(14, dtype=np.float32)
    )

    print(f"probing {url}")
    encoded_ok = False
    for quality, label in ((DEFAULT_MOLMOACT_JPEG_QUALITY, "jpeg q95"), (0, "raw")):
        client = MolmoActClient(url, jpeg_quality=quality, timeout_s=args.timeout)
        try:
            actions = client.infer(observation, args.instruction)
            latency = client.last_latency or {}
            print(
                f"  {label:9} ok   {actions.shape[0]} actions  "
                f"{latency.get('round_trip_s', 0.0):.2f}s  "
                f"{latency.get('payload_mb', 0.0):.2f} MB"
            )
            encoded_ok = encoded_ok or quality > 0
        except EncodedFramesUnsupported as err:
            print(f"  {label:9} RAW-ONLY SERVER\n             {err}")
        except InferenceError as err:
            print(f"  {label:9} failed: {err}", file=sys.stderr)
        finally:
            client.close()

    if encoded_ok:
        print("\nthis server decodes encoded frames; nothing to do")
        return 0
    print(
        "\nthis server is raw-only. Patch its _to_pil in place -- do not copy "
        "host_server_yam.py over it:\n"
        "  scp scripts/patch_server_to_pil.py USER@HOST:/tmp/\n"
        "  ssh USER@HOST 'python3 /tmp/patch_server_to_pil.py "
        "~/molmoact2/examples/yam/host_server_yam.py'\n"
        "then restart the server. Until then every request pays ~14x the payload."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
