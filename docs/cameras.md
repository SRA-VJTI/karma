# Cameras

The bimanual cell watches itself with three Intel RealSense D405s: one overhead
and one on each wrist. They belong to the rig, exactly like the CAN interfaces
do, so `doctor --rig`, `cameras`, and anything that records agree on what
`left_wrist` means.

```bash
uv sync
uv run openpi-control cameras                # what is plugged in, and where
uv run openpi-control cameras --probe        # also open each one and measure it
```

## A camera's identity is its serial

A `/dev/videoN` number is not a camera. It changes with boot order, with which
USB port you used, and with how many cameras came up first. So a rig pins each
view by serial number, and `openpi_control.cameras` resolves that to a device
at run time:

| Camera | Model | Serial | Capture | Rides on | Sees |
| --- | --- | --- | --- | --- | --- |
| `top` | D435 | `348523020354` | 640x480@30 | — | the whole cell, from above |
| `left_wrist` | D405 | `254623070863` | 848x480@30 | `left` | what the left gripper is about to touch |
| `right_wrist` | D405 | `254623070417` | 848x480@30 | `right` | the same, for the right arm |

The wrist serials are the ones the cell was already using — they match
`cams.env` in `vr-teleop-kit`. Swap a camera and the one place to edit is
`YAM_BIMANUAL_CAMERA_SERIALS` in `openpi_control/rigs.py`.

### The top camera is a D435, and it is not interchangeable

It replaced a D405 (`254623070531`), and it differs in two ways the rig has to
carry rather than paper over:

- **It publishes no serial in its USB descriptor.** Its udev names end in the
  model number where a D405's end in the serial, and its colour node gets no
  `/dev/v4l/by-id` entry at all. Discovery therefore falls back to
  `sdk_present_asic_serials()` for cameras udev cannot name; the SDK addresses
  it perfectly well, and device paths here are only ever diagnostics — a stream
  is opened by serial. Such a camera shows up as `sdk:<sdk-serial>` in the
  `cameras` table instead of a `/dev/...` path.
- **It came up on USB 2.0, and captures 640x480@30 because of it.** Enumerated
  on 2026-08-23 it offered only 424x240, 640x480, 1280x720@15 and 1920x1080@8 —
  the reduced set a D435 falls back to on USB 2. A USB 3 D435 also has
  848x480@30, which is the rig default and what the wrists run. So
  `YAM_TOP_CAPTURE` in `rigs.py` is a workaround with an expiry date: re-seat
  the camera on a USB 3 port, re-run `openpi-control cameras`, and delete the
  override once 848x480@30 shows up. All three cameras do hold a full 30 fps
  together as configured.

  The aspect ratio is the reason to bother. MolmoAct2's training frames are
  640x360 and the D405 wrists are 848x480 — both 16:9. 640x480 is 4:3, so the
  top view is currently the one input whose shape does not match what the
  policy was trained on.

The checkpoint itself was trained with a D435 in the top role, so the camera
*model* is a move toward the training setup rather than away from it — note
that the reference deployment on this cell had a D405 there
(`front_camera: 352122273221` in `~/molmoact2/examples/yam/configs/yam_left.yaml`,
which is the SDK serial for ASIC `254623070531`), so the reference is not the
authority on this one.

What the swap does not fix: `YAM_TOP_CAMERA_EXTRINSIC` still describes where
the old camera sat, so the top-camera frame in the Viser scene is wrong until
it is recalibrated. That is visualization metadata only — no capture or policy
path reads it.

Naming the arm a wrist camera rides on is what makes `--only` do the obviously
right thing:

```bash
uv run openpi-control cameras --only right
# top + right_wrist. The left wrist camera goes wherever the left arm goes.
```

Recording a left-wrist view of an arm that was never powered would put the same
frozen frame in every step of the dataset, so the camera is dropped rather than
left in.

### The two serial numbers of a D405

A D405 answers to two different numbers, which is worth knowing before you go
looking for one of them:

| Number | Example | Where you see it |
| --- | --- | --- |
| ASIC serial | `254623070531` | `/dev/v4l/by-id`, the USB descriptor, `cams.env`, this rig |
| SDK serial | `352122273221` | `pyrealsense2`'s `serial_number`, and nothing else |

Rigs declare the ASIC serial, because that is the one you can look up with
`ls /dev/v4l/by-id` on a box with no SDK installed. `sdk_serial_for_asic()`
bridges to the other one when a stream actually opens. If you ever see "no
RealSense with ASIC serial … is connected" while `rs-enumerate-devices` clearly
lists three cameras, this is why.

## Discovery needs nothing installed

Answering "is the top camera plugged in" is pure filesystem work — glob
`/dev/v4l/by-id`, read the serial out of each entry's name, keep the colour
node. No SDK, no OpenCV. That is deliberate: `doctor` has to be able to tell you
a camera fell off its mount on a machine that cannot open a camera at all.

```
uv run openpi-control doctor --rig yam_bimanual
...
cameras (3 declared):
  [OK  ] camera top                 848x480@30 — /dev/v4l/by-id/usb-Intel_R_...-video-index4
  [WARN] camera left_wrist          serial 254623070863 not on the bus
  [OK  ] camera right_wrist         848x480@30 — /dev/v4l/by-id/usb-Intel_R_...-video-index4
```

A missing camera is a warning here, not a failure: none of them are needed to
drive an arm, and `doctor` should not refuse to green-light a cell because
someone unplugged a wrist camera to work on it. A recorder asks the same
function with `required=True`, where a missing view really is fatal.

A camera that is on the bus but in no rig is reported too — that is usually the
informative half of "the top view is missing":

```
  [WARN] unclaimed cameras          serial(s) 254623070987 present but in no rig
                                    camera — add them to YAM_BIMANUAL_CAMERA_SERIALS
```

## `--probe` opens them

`--probe` starts each stream, waits for a frame, lets it settle, then counts
frames for two seconds. `--snapshot DIR` writes what it grabbed, which is the
fastest way to check where a wrist camera is actually pointing:

```bash
uv run openpi-control cameras --probe --snapshot /tmp/cams
```

```
  [OK  ] probe top                  848x480 bgr8, 30 fps (asked 30) -> /tmp/cams/top.png
  [OK  ] probe left_wrist           848x480 bgr8, 30 fps (asked 30) -> /tmp/cams/left_wrist.png
  [OK  ] probe right_wrist          848x480 bgr8, 30 fps (asked 30) -> /tmp/cams/right_wrist.png
```

Cameras are opened one at a time, so the useful error — "this camera is held by
something else" — is not hidden behind two others complaining about USB
bandwidth. A camera can only be streamed by one process at a time: a recorder
and a browser preview cannot both have it.

## In the browser

`live` puts the same cameras on the viser page as the arms, one tile each under
**Cameras**, which is the fastest way to check where a wrist is pointing while
you drive it:

```bash
uv run openpi-control live --rig yam_bimanual --control
uv run openpi-control live --no-cameras       # leave them free for a recorder
```

A tile is a preview, not a recording: 400 px wide at 10 Hz, against the 30 Hz
the poses go out at. Pushing three 848x480 streams whole on the mirror clock
would be ~35 MB/s of websocket to answer a question a thumbnail answers. A
camera that is unplugged, or held by another process, is named on stdout and
simply gets no tile — see [Cameras in the browser](cli.md#cameras-in-the-browser).

`CameraPanel` in `openpi_control.viz` is the panel itself, and it takes readers
that are *already open* — it never opens one. That is what keeps `viz` drawing
and holding no device, so it stays importable on a box with no RealSense SDK:

```python
from openpi_control.viz import CameraPanel

panel = CameraPanel(scene.server, readers)   # readers from open_readers()
while running:
    panel.step(dt)                           # throttles itself internally
```

## Two measurements worth knowing

Both of these were measured on this cell, and both are the reason the defaults
are what they are.

**848x480, not 640x480.** 848x480 is the D405's native colour mode. Asking for
640x480 makes the firmware rescale, and that is not free:

| Mode | Three cameras at once |
| --- | --- |
| 848x480 | 30, 30, 29.5 fps |
| 640x480 | 20, 17, 15 fps |

If a policy wants a different size, crop or resize downstream — far cheaper
there than in the camera.

**848x480 also runs at 90 fps.** Every mode up to 848x480 offers 5/15/30/60/90;
only 1280x720 caps at 30. All three cameras hold a true 90 fps concurrently
(~220 MB/s over the shared USB 3 uplink, no drops). The rig declares 30 because
that is the sane default for a dataset, and a run overrides it —
`Rig.with_camera_capture(fps=..., pixel_format=...)`, which is what
`openpi-control record --fps 90` does. `cameras.supported_color_modes(serial)`
lists what a given camera offers without opening a stream.

**The SDK, not OpenCV.** Capture goes through `pyrealsense2`. Reading the same
colour node through `cv2.VideoCapture` tops out around 10–13 fps, while
`v4l2-ctl` streams that node at a clean 30 — so the ceiling is in OpenCV's UVC
consumer, not in the camera, the cable, or the bus (all three cameras are on
USB 3 links here). Through the SDK all three hold a full 90 fps concurrently.
OpenCV is still a dependency, but only to encode snapshots.

If you record through some other tool, check its real frame rate before
trusting the fps in its metadata.

## Overriding a camera

`--camera NAME=DEVICE` pins one camera to an explicit device, the camera
equivalent of `--interface ARM=IFACE`:

```bash
uv run openpi-control cameras --camera top=/dev/video4
```

A pinned path that does not exist is reported as missing rather than quietly
falling back to whatever discovery found — an operator who names a device meant
that one. A name that is not in the rig is an error, not a no-op.

Two per-camera fields exist for mechanical facts rather than flags:

| Field | For |
| --- | --- |
| `rotate` (0/90/180/270) | a camera mounted sideways; applied in the capture thread so every consumer sees the corrected frame |
| `color_index` | a RealSense model whose colour stream is not on v4l2 node 4 (it is, on a D405) |

## Reading frames yourself

```python
from openpi_control.cameras import discover, open_readers, close_readers
from openpi_control.rigs import resolve_rig

rig = resolve_rig("yam_bimanual")
found = discover(rig.cameras)
if not found.complete:
    raise SystemExit(f"not on the bus: {found.missing}")

readers = open_readers(found.specs())
try:
    for name, reader in readers.items():
        frame = reader.wait_for_frame()      # BGR, HxWx3 uint8
        print(name, frame.shape, reader.negotiated)
finally:
    close_readers(readers)
```

`latest()` is latest-frame-wins: a control loop wants the freshest image at the
moment it asks, never a backlog that grows whenever the consumer falls behind.
It returns `None` until the first frame lands, which is what `wait_for_frame`
is for — start writing before then and every episode gets a hole at the front.

`open_readers` opens all of them or none. A half-open camera set is how you end
up with a dataset that is quietly missing a view.
