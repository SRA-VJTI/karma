"""The parts of recording that only a real LeRobotDataset can answer.

Everything in test_record.py runs against a fake sink, which is what makes the
loop testable at all. These tests are the other half: they write real datasets
to a temp dir and read them back, because the questions here are about what
LeRobot actually does, not about what this package asks it to do -- and the
answers have changed between LeRobot versions.

Skipped when ``lerobot`` is not importable, which is the normal state on a robot
box (the extra needs Python 3.12+ and pulls torch).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="needs the lerobot extra")

from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    CODEBASE_VERSION,
    LeRobotDataset,
)

from openpi_control.record import (  # noqa: E402
    DATASET_GRIPPER_CLOSED,
    DATASET_GRIPPER_OPEN,
    LeRobotSink,
    build_features,
)

STATE_NAMES = ["left_joint_1", "left_gripper"]
SHAPE = (32, 32, 3)


def sink(tmp_path, *, fps: int = 30, cameras: bool = True) -> LeRobotSink:
    return LeRobotSink(
        repo_id="local/test",
        fps=fps,
        features=build_features(STATE_NAMES, {"top": SHAPE} if cameras else {}),
        robot_type="test_rig",
        root=tmp_path / "ds",
        image_writer_threads=4,
    )


def frame(
    joint: float, gripper: float, *, red: int = 0, camera: bool = True
) -> dict[str, object]:
    """One row. ``red`` tags the image so frames can be told apart.

    LeRobot rejects a frame whose keys do not match the schema exactly, so a
    dataset created without cameras needs rows without the image key.
    """
    row: dict[str, object] = {
        "observation.state": np.array([joint, gripper], dtype=np.float32),
        "action": np.array([joint, gripper], dtype=np.float32),
        "task": "a test task",
    }
    if camera:
        image = np.zeros(SHAPE, dtype=np.uint8)
        image[..., 0] = red
        row["observation.images.top"] = image
    return row


def files(root) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_a_recorded_dataset_is_lerobot_v3(tmp_path) -> None:
    # The format is the deliverable: a v2 dataset would not load in the tooling
    # this whole path exists to feed.
    writer = sink(tmp_path)
    for index in range(4):
        writer.add_frame(frame(0.1 * index, DATASET_GRIPPER_OPEN))
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("local/test", root=tmp_path / "ds")

    assert CODEBASE_VERSION == "v3.0"
    assert dataset.meta.info["codebase_version"] == "v3.0"
    assert dataset.meta.info["robot_type"] == "test_rig"
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 4
    assert dataset.fps == 30


def test_cameras_are_stored_as_video_not_a_pile_of_images(tmp_path) -> None:
    # `video` is the reason an hour of three cameras is gigabytes and not
    # terabytes. It also means the on-disk layout has mp4s, not PNG dirs.
    writer = sink(tmp_path)
    for _ in range(4):
        writer.add_frame(frame(0.0, DATASET_GRIPPER_OPEN))
    writer.save_episode()
    writer.finalize()

    on_disk = files(tmp_path / "ds")

    assert any(name.endswith(".mp4") for name in on_disk)
    assert any(name.endswith(".parquet") for name in on_disk)
    assert not [name for name in on_disk if name.endswith(".png")]


def test_svt_encoder_is_quiet_and_uses_a_supported_preset(
    tmp_path, monkeypatch, capfd
) -> None:
    # SVT bypasses Python/FFmpeg log levels and writes its banner directly to
    # stderr.  Capture at the file-descriptor level so this test sees the same
    # native output that otherwise floods an operator's terminal.
    monkeypatch.delenv("SVT_LOG", raising=False)
    writer = sink(tmp_path)

    assert os.environ["SVT_LOG"] == "1"  # errors only
    assert writer._dataset.writer._rgb_encoder.preset == 10

    for _ in range(2):
        writer.add_frame(frame(0.0, DATASET_GRIPPER_OPEN))
    writer.save_episode()
    writer.finalize()

    assert "Svt[" not in capfd.readouterr().err


def test_a_discarded_take_leaves_nothing_behind(tmp_path) -> None:
    # The failure this guards against is silent and expensive: a discarded take
    # whose frames end up inside the next saved episode. It is one call on
    # lerobot 3.0 and was not on older versions, so assert on the dataset that
    # comes back rather than on the API that was called.
    writer = sink(tmp_path)
    root = tmp_path / "ds"

    for _ in range(5):
        writer.add_frame(frame(1.0, DATASET_GRIPPER_CLOSED, red=11))
    writer.discard_episode()

    assert writer.num_episodes == 0
    # Nothing staged and nothing left over. With streaming encoding there are no
    # temporary PNGs to begin with; on the staged path the discard is what
    # removes them. Either way the answer here is the same.
    assert not [name for name in files(root) if name.endswith(".png")]

    for _ in range(3):
        writer.add_frame(frame(2.0, DATASET_GRIPPER_OPEN, red=200))
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("local/test", root=root)
    assert dataset.num_frames == 3  # not 8
    joints = [float(dataset[i]["observation.state"][0]) for i in range(3)]
    assert joints == [2.0, 2.0, 2.0]  # no 1.0 from the thrown-away take
    assert not [name for name in files(root) if name.endswith(".png")]


def test_the_staged_path_also_discards_cleanly(tmp_path) -> None:
    # streaming_encoding=False is the older behaviour: every frame is staged as
    # a PNG and encoded inside save_episode. It stays reachable, so the discard
    # has to clean up after it too.
    writer = LeRobotSink(
        repo_id="local/staged",
        fps=30,
        features=build_features(STATE_NAMES, {"top": SHAPE}),
        robot_type="test_rig",
        root=tmp_path / "ds",
        image_writer_threads=2,
        streaming_encoding=False,
    )
    root = tmp_path / "ds"

    for _ in range(4):
        writer.add_frame(frame(1.0, DATASET_GRIPPER_CLOSED, red=11))

    writer.discard_episode()

    # Whether the async writer had flushed a PNG yet is a race, so this asserts
    # the part that is deterministic: after the discard there is nothing staged,
    # whether or not anything got there first.
    assert not [name for name in files(root) if name.endswith(".png")]
    assert writer.num_episodes == 0


def test_an_empty_episode_cannot_be_saved(tmp_path) -> None:
    # The record loop relies on this being an error rather than a silent no-op:
    # it is why a save pressed before any frame landed keeps the take open.
    writer = sink(tmp_path)

    with pytest.raises(Exception):  # noqa: B017 - lerobot's own error type
        writer.save_episode()


def test_the_gripper_column_round_trips_in_dataset_units(tmp_path) -> None:
    # 0.0 open, 1.0 closed, with intermediate analog values preserved all the
    # way through parquet. If this inverts or quantizes, policies cannot learn
    # how far to close the gripper.
    writer = sink(tmp_path, cameras=False)
    writer.add_frame(frame(0.0, DATASET_GRIPPER_OPEN, camera=False))
    writer.add_frame(frame(0.0, 0.37, camera=False))
    writer.add_frame(frame(0.0, DATASET_GRIPPER_CLOSED, camera=False))
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("local/test", root=tmp_path / "ds")

    assert float(dataset[0]["observation.state"][1]) == pytest.approx(0.0)
    assert float(dataset[1]["observation.state"][1]) == pytest.approx(0.37)
    assert float(dataset[1]["action"][1]) == pytest.approx(0.37)
    assert float(dataset[2]["observation.state"][1]) == pytest.approx(1.0)


def test_the_state_and_action_column_names_survive(tmp_path) -> None:
    # These names are the dataset's contract with whatever trains on it.
    writer = sink(tmp_path, cameras=False)
    writer.add_frame(frame(0.0, DATASET_GRIPPER_OPEN, camera=False))
    writer.save_episode()
    writer.finalize()

    info = LeRobotDataset("local/test", root=tmp_path / "ds").meta.info

    assert info["features"]["observation.state"]["names"] == STATE_NAMES
    assert info["features"]["action"]["names"] == STATE_NAMES


def test_a_high_rate_dataset_records_its_real_rate(tmp_path) -> None:
    # fps is metadata, and metadata that disagrees with the frames is how a
    # dataset silently becomes untrainable. 90 is the rate this cell can hold.
    writer = sink(tmp_path, fps=90)
    for _ in range(4):
        writer.add_frame(frame(0.0, DATASET_GRIPPER_OPEN))
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("local/test", root=tmp_path / "ds")

    assert dataset.fps == 90
    # Timestamps must step at the declared rate, not at 30.
    steps = np.diff([float(dataset[i]["timestamp"]) for i in range(4)])
    assert steps == pytest.approx([1 / 90] * 3, abs=1e-6)
