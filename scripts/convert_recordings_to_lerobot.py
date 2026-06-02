#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "lerobot[dataset]>=0.5.1",
#   "mcap>=1.3.1",
#   "mcap-protobuf-support>=0.5.4",
#   "imageio[ffmpeg]>=2.34",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Convert a karma `recordings/` tree into a MolmoAct2-BimanualYAM compatible
LeRobotDataset.

Input layout (produced by Karma's Session/recording.py):
    recordings/[task_name/]YYYYMMDD/episode_HHMMSS_<uid>/
        yam_left.mcap        # RobotNode -> /yam_left/joint_state (JSON, joint_pos[6] + gripper_pos[1])
        yam_right.mcap       # RobotNode -> /yam_right/joint_state
        gello_left.mcap      # AgentNode -> /gello_left/joint_pos (JSON, joint_pos[7] incl. gripper)
        gello_right.mcap     # AgentNode -> /gello_right/joint_pos
        camera_top-images-rgb.mp4    + camera_top-rgb-timestamp.npy
        camera_left-images-rgb.mp4   + camera_left-rgb-timestamp.npy
        camera_right-images-rgb.mp4  + camera_right-rgb-timestamp.npy
        session_meta.json

Node names are configurable via --state-nodes / --action-nodes; defaults
above match the bimanual gello teleop config that produced the current
recordings. The episode walker handles arbitrary nesting between the
provided --recordings-root and the `episode_*` leaf dirs.

Output:
    Standard LeRobot v3.0 dataset with:
        observation.state          (14,)   float32  [left_arm_7, right_arm_7]
        observation.images.top     (H,W,3) video
        observation.images.left    (H,W,3) video
        observation.images.right   (H,W,3) video
        action                     (14,)   float32  [leader_left_7, leader_right_7]
        task                       string  (per-episode language instruction)
    fps = 10  (matches MolmoAct2 server's control rate and norm_stats)
    norm_tag (downstream) = "yam_dual_molmoact2"

Run from a venv with `lerobot[dataset]`, `mcap`, `mcap-protobuf-support`, and
`imageio[ffmpeg]` installed, e.g.:

    uv run --script karma/scripts/convert_recordings_to_lerobot.py \\
        --recordings-root /home/sra/yam/karma/recordings \\
        --output-root /home/sra/lerobot_data/local/yam-stack-cubes \\
        --repo-id local/yam-stack-cubes \\
        --task "stack the cubes"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from mcap.reader import make_reader
from tqdm import tqdm


# ---- Stream readers ----------------------------------------------------------


@dataclass
class TimedSamples:
    """Two parallel arrays: timestamps (seconds) and values (np.ndarray rows)."""
    ts: np.ndarray  # shape (N,) float64
    val: np.ndarray  # shape (N, D) float32

    def sample_at(self, t: float) -> np.ndarray:
        """Return the most-recent sample at-or-before t (or the first if t precedes start)."""
        idx = int(np.searchsorted(self.ts, t, side="right") - 1)
        if idx < 0:
            idx = 0
        return self.val[idx]


def _iter_json_messages(path: Path, topic_suffix: str):
    """Yield (log_time_sec, payload_dict) for every JSON message on /*/{topic_suffix}."""
    with open(path, "rb") as fh:
        reader = make_reader(fh)
        for schema, channel, message in reader.iter_messages():
            if not channel.topic.endswith(f"/{topic_suffix}"):
                continue
            if schema is None or schema.encoding != "jsonschema":
                continue
            try:
                payload = json.loads(message.data.decode("utf-8"))
            except Exception:
                continue
            yield message.log_time / 1e9, payload


def _read_joint_state_mcap(path: Path, arm_dof: int = 6) -> TimedSamples:
    """Read `/{yam_*}/joint_state` JSON messages and concat joint_pos + gripper_pos.

    Returns TimedSamples with val shape (N, arm_dof+1) — the 7-D per-arm state
    vector the MolmoAct2-BimanualYAM model expects.
    """
    ts: list[float] = []
    vals: list[list[float]] = []
    for t, payload in _iter_json_messages(path, "joint_state"):
        jp = payload.get("joint_pos")
        gp = payload.get("gripper_pos")
        if jp is None or gp is None:
            continue
        jp_l = list(jp)[:arm_dof]
        gp_l = list(gp) if isinstance(gp, (list, tuple)) else [float(gp)]
        if len(jp_l) != arm_dof or len(gp_l) != 1:
            continue
        ts.append(t)
        vals.append(jp_l + gp_l)
    if not ts:
        raise RuntimeError(f"{path}: no /joint_state JSON messages found")
    arr_ts = np.asarray(ts, dtype=np.float64)
    arr_val = np.asarray(vals, dtype=np.float32)
    order = np.argsort(arr_ts)
    return TimedSamples(ts=arr_ts[order], val=arr_val[order])


def _read_action_mcap(path: Path, expected_dim: int = 7) -> TimedSamples:
    """Read `/{gello_*}/joint_pos` JSON messages; payload joint_pos already includes gripper."""
    ts: list[float] = []
    vals: list[list[float]] = []
    for t, payload in _iter_json_messages(path, "joint_pos"):
        jp = payload.get("joint_pos") or payload.get("pos") or payload.get("position")
        if jp is None:
            continue
        jp_l = list(jp)[:expected_dim]
        if len(jp_l) != expected_dim:
            continue
        ts.append(t)
        vals.append(jp_l)
    if not ts:
        raise RuntimeError(f"{path}: no /joint_pos JSON messages found")
    arr_ts = np.asarray(ts, dtype=np.float64)
    arr_val = np.asarray(vals, dtype=np.float32)
    order = np.argsort(arr_ts)
    return TimedSamples(ts=arr_ts[order], val=arr_val[order])


def _read_video(mp4: Path, ts_npy: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames, timestamps) where frames is (N, H, W, 3) uint8."""
    ts = np.load(ts_npy)
    reader = imageio.get_reader(str(mp4))
    frames = [np.asarray(f, dtype=np.uint8) for f in reader]
    reader.close()
    if len(frames) != len(ts):
        # In rare cases the encoder drops the trailing frame at sentinel; align by truncation.
        n = min(len(frames), len(ts))
        frames = frames[:n]
        ts = ts[:n]
    return np.stack(frames, axis=0), np.asarray(ts, dtype=np.float64)


def _resample_index(stream_ts: np.ndarray, target_ts: float) -> int:
    idx = int(np.searchsorted(stream_ts, target_ts, side="right") - 1)
    return max(idx, 0)


# ---- Episode discovery -------------------------------------------------------


@dataclass
class EpisodePaths:
    root: Path
    state_left: Path  # yam_left.mcap
    state_right: Path  # yam_right.mcap
    action_left: Path  # gello_left.mcap
    action_right: Path  # gello_right.mcap
    cam_top_mp4: Path
    cam_top_ts: Path
    cam_left_mp4: Path
    cam_left_ts: Path
    cam_right_mp4: Path
    cam_right_ts: Path


def _build_episode_paths(
    ep_dir: Path,
    state_nodes: tuple[str, str],
    action_nodes: tuple[str, str],
) -> EpisodePaths:
    sl, sr = state_nodes
    al, ar = action_nodes
    return EpisodePaths(
        root=ep_dir,
        state_left=ep_dir / f"{sl}.mcap",
        state_right=ep_dir / f"{sr}.mcap",
        action_left=ep_dir / f"{al}.mcap",
        action_right=ep_dir / f"{ar}.mcap",
        cam_top_mp4=ep_dir / "camera_top-images-rgb.mp4",
        cam_top_ts=ep_dir / "camera_top-rgb-timestamp.npy",
        cam_left_mp4=ep_dir / "camera_left-images-rgb.mp4",
        cam_left_ts=ep_dir / "camera_left-rgb-timestamp.npy",
        cam_right_mp4=ep_dir / "camera_right-images-rgb.mp4",
        cam_right_ts=ep_dir / "camera_right-rgb-timestamp.npy",
    )


def _find_episodes(
    recordings_root: Path,
    state_nodes: tuple[str, str],
    action_nodes: tuple[str, str],
) -> list[EpisodePaths]:
    """Walk recordings_root for any `episode_*` dir, regardless of nesting depth."""
    eps: list[EpisodePaths] = []
    if not recordings_root.exists():
        return eps
    for ep_dir in sorted(recordings_root.rglob("episode_*")):
        if not ep_dir.is_dir():
            continue
        paths = _build_episode_paths(ep_dir, state_nodes, action_nodes)
        missing = [
            getattr(paths, f).name
            for f in paths.__dataclass_fields__
            if f != "root" and not getattr(paths, f).exists()
        ]
        if missing:
            logging.warning("skipping %s (missing: %s)", ep_dir.relative_to(recordings_root), missing)
            continue
        eps.append(paths)
    return eps


# ---- Conversion --------------------------------------------------------------


def _convert_episode(ep: EpisodePaths, dataset, task: str, fps: int, dim_per_arm: int = 7) -> int:
    """Add a single episode to the LeRobotDataset; returns frames written."""
    arm_dof = dim_per_arm - 1  # last element is gripper
    left_state = _read_joint_state_mcap(ep.state_left, arm_dof=arm_dof)
    right_state = _read_joint_state_mcap(ep.state_right, arm_dof=arm_dof)
    leader_left = _read_action_mcap(ep.action_left, expected_dim=dim_per_arm)
    leader_right = _read_action_mcap(ep.action_right, expected_dim=dim_per_arm)

    cam_top_frames, cam_top_ts = _read_video(ep.cam_top_mp4, ep.cam_top_ts)
    cam_left_frames, cam_left_ts = _read_video(ep.cam_left_mp4, ep.cam_left_ts)
    cam_right_frames, cam_right_ts = _read_video(ep.cam_right_mp4, ep.cam_right_ts)

    # Active window = intersection of all streams.
    t_start = max(
        left_state.ts[0],
        right_state.ts[0],
        leader_left.ts[0],
        leader_right.ts[0],
        cam_top_ts[0],
        cam_left_ts[0],
        cam_right_ts[0],
    )
    t_end = min(
        left_state.ts[-1],
        right_state.ts[-1],
        leader_left.ts[-1],
        leader_right.ts[-1],
        cam_top_ts[-1],
        cam_left_ts[-1],
        cam_right_ts[-1],
    )
    if t_end - t_start < 1.0 / fps:
        logging.warning("skipping %s — active window <1 tick", ep.root.name)
        return 0

    n_frames = int(np.floor((t_end - t_start) * fps)) + 1
    grid = t_start + np.arange(n_frames) / fps

    for t in grid:
        state = np.concatenate([left_state.sample_at(t), right_state.sample_at(t)]).astype(
            np.float32
        )
        action = np.concatenate([leader_left.sample_at(t), leader_right.sample_at(t)]).astype(
            np.float32
        )
        top = cam_top_frames[_resample_index(cam_top_ts, t)]
        left = cam_left_frames[_resample_index(cam_left_ts, t)]
        right = cam_right_frames[_resample_index(cam_right_ts, t)]

        dataset.add_frame(
            {
                "observation.state": state,
                "action": action,
                "observation.images.top": top,
                "observation.images.left": left,
                "observation.images.right": right,
                "task": task,
            }
        )

    dataset.save_episode()
    return n_frames


def _features_dict(img_top_shape, img_left_shape, img_right_shape, dim: int = 14) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": {
                "axes": [
                    *(f"left_joint_{i}" for i in range(dim // 2 - 1)),
                    "left_gripper",
                    *(f"right_joint_{i}" for i in range(dim // 2 - 1)),
                    "right_gripper",
                ],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": {
                "axes": [
                    *(f"left_joint_{i}" for i in range(dim // 2 - 1)),
                    "left_gripper",
                    *(f"right_joint_{i}" for i in range(dim // 2 - 1)),
                    "right_gripper",
                ],
            },
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": img_top_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.left": {
            "dtype": "video",
            "shape": img_left_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.right": {
            "dtype": "video",
            "shape": img_right_shape,
            "names": ["height", "width", "channels"],
        },
    }


def _peek_image_shape(mp4: Path) -> tuple[int, int, int]:
    reader = imageio.get_reader(str(mp4))
    try:
        frame = reader.get_next_data()
    finally:
        reader.close()
    arr = np.asarray(frame)
    return (int(arr.shape[0]), int(arr.shape[1]), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Local root for the LeRobotDataset (passed as `root=` to create()).")
    ap.add_argument("--repo-id", default="local/yam-stack-cubes",
                    help="LeRobotDataset repo_id. Used only for metadata; no hub push.")
    ap.add_argument("--task", required=True,
                    help='Language instruction recorded with every frame (e.g. "stack the cubes").')
    ap.add_argument("--fps", type=int, default=10,
                    help="Output dataset control rate. 10 matches MolmoAct2 server/norm_stats.")
    ap.add_argument("--robot-type", default="bimanual_yam")
    ap.add_argument("--use-videos", action="store_true", default=True,
                    help="Store images as MP4 videos (default).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of episodes converted.")
    ap.add_argument("--state-nodes", nargs=2, metavar=("LEFT", "RIGHT"),
                    default=["yam_left", "yam_right"],
                    help="MCAP basenames (no .mcap) for left/right state nodes.")
    ap.add_argument("--action-nodes", nargs=2, metavar=("LEFT", "RIGHT"),
                    default=["gello_left", "gello_right"],
                    help="MCAP basenames (no .mcap) for left/right action nodes (teleop leaders).")
    ap.add_argument("--vcodec", default="libsvtav1",
                    help="Video codec for LeRobotDataset.create. 'h264' (libx264, fast CPU), "
                         "'h264_nvenc' (GPU), 'libsvtav1' (default, slow AV1), or 'auto'.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eps = _find_episodes(
        args.recordings_root,
        state_nodes=tuple(args.state_nodes),  # type: ignore[arg-type]
        action_nodes=tuple(args.action_nodes),  # type: ignore[arg-type]
    )
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        logging.error("no episodes found under %s", args.recordings_root)
        return 1
    logging.info("discovered %d episodes", len(eps))

    img_top_shape = _peek_image_shape(eps[0].cam_top_mp4)
    img_left_shape = _peek_image_shape(eps[0].cam_left_mp4)
    img_right_shape = _peek_image_shape(eps[0].cam_right_mp4)
    logging.info(
        "image shapes — top=%s left=%s right=%s", img_top_shape, img_left_shape, img_right_shape
    )

    features = _features_dict(img_top_shape, img_left_shape, img_right_shape, dim=14)

    # Import LeRobotDataset late so --help works without the heavy install.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobotDatasetMetadata.create insists output_root not pre-exist.
    if args.output_root.exists():
        logging.error("output-root already exists: %s (delete it or pick another path)", args.output_root)
        return 2
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_root,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
        vcodec=args.vcodec,
    )

    n_total = 0
    for ep in tqdm(eps, desc="episodes"):
        try:
            n = _convert_episode(ep, dataset, task=args.task, fps=args.fps)
        except Exception as exc:  # noqa: BLE001
            logging.exception("FAILED on %s: %s", ep.root, exc)
            continue
        logging.info("%s -> %d frames", ep.root.name, n)
        n_total += n

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    logging.info("done. %d frames across %d episodes written to %s", n_total, len(eps), args.output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
