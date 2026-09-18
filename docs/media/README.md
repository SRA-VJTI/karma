# Documentation media

The README uses looping GIF previews linked to full MP4 clips. Previews animate
on page load; full videos are opened on demand. Detail guides use lightweight
JPEG posters. The main VR demo links to the
author's X post: https://x.com/k7agar/status/2094818716433465631.

Raw originals live in the Git-ignored `media-inbox/` directory.

| Original | Published asset |
| --- | --- |
| `vr_teleop_setup.JPG` | `quest-camera-views.jpg` |
| `so101_teleop.MOV` | `so101-quest-teleop.mp4` |
| `molmaoct2_so101_rollout.mp4` | `so101-molmoact2-rollout.mp4` |
| `yam_human_in_the_loop.MOV` | `yam-human-in-the-loop.mp4` |

Videos retain their original duration and 30 fps. Exports use H.264, yuv420p,
CRF 26, the slow preset, and MP4 fast-start metadata.
Audio and source metadata are omitted from these documentation copies.
Phone HDR footage is tone-mapped to BT.709 SDR with FFmpeg's zscale and Hable
tonemap filters; use an FFmpeg build containing zscale (such as ffmpeg-full).

SO101 teleop is cropped after automatic portrait rotation to 1080×1080 at
(0, 460) and exported at 640×640. YAM uses a wide 1440×900 crop at (0, 180),
exported at 960×600, preserving both arms while excluding the operator's face
at the right of the source frame.
These crops apply to the MP4s, GIFs and JPEG posters. Originals remain untouched.
The rollout retains its 848×478 frame. SO101 teleop and YAM posters are
320×320 at 3s and 480×300 at 23s respectively; the rollout poster is 480×270 at 17s.

The hero GIF uses cropped YAM footage from 19–25s at 640×400, 8 fps and
1.25× speed. Gallery GIFs use 8 fps: SO101 teleop from 2–8s at 280×280 and
YAM from 19–26s at 320×200, both at 1.25× speed; rollout from 11–20s at
320×180 and 1.5× speed.
GIFs loop indefinitely and are optimized with `gifsicle -O3 --lossy=60`.
The Quest camera-view screenshot remains in the camera guide.

## Branding

The supplied logo is preserved in `media-inbox/karma-logo-original.png`.
`karma-banner.png` is a centered 1280×448 crop at (384, 138) for the README.
`karma-social-preview.png` is a 1280×640 crop at (384, 42) for GitHub's
repository social preview. Both preserve the supplied artwork without redrawing it.
