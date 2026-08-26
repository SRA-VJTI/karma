"""The fake ``/dev/v4l/by-id`` bus, shared by the camera tests.

Lives here rather than in ``test_cameras.py`` so that ``test_cli.py`` can build
the same bus without importing another test module.
"""

from __future__ import annotations

# The real udev name shape, so a change to the parsing regex is caught by a
# name in the form udev actually produces rather than a simplified stand-in.
_BY_ID_TEMPLATE = (
    "usb-Intel_R__RealSense_TM__Depth_Camera_405_"
    "Intel_R__RealSense_TM__Depth_Camera_405_{serial}-video-index{index}"
)

# The six v4l2 nodes a D405 publishes, so the fake bus looks like the real one.
ALL_NODES = (0, 1, 2, 3, 4, 5)


def fake_by_id(tmp_path, serials, *, indices=ALL_NODES):
    """A stand-in /dev/v4l/by-id holding all six nodes of each camera."""
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    for serial in serials:
        for index in indices:
            (by_id / _BY_ID_TEMPLATE.format(serial=serial, index=index)).touch()
    return by_id
