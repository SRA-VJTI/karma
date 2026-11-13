"""Flash-style YAM ACT contract, shared operator masking, and chunk transitions.

This is a YAM adaptation, not an ARX checkpoint loader. Raw recordings stay intact;
exactly the same RGB mask/resize/normalization runs in training and serving.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .exceptions import ConfigurationError
from .inference import InferenceError, MolmoActClient, PolicyContract

TAG = "yam_flash_act_v1"
CAMERAS = ("top", "left_wrist", "right_wrist")
CONTRACT = PolicyContract(norm_tag=TAG)


@dataclass(frozen=True)
class FlashConfig:
    schema: str = TAG
    fps: int = 30
    image_size: tuple[int, int] = (240, 320)  # height, width
    masks: dict[str, list[list[float]]] = field(default_factory=dict)
    chunk_size: int = 50
    execute_steps: int = 25
    blend_steps: int = 10
    task: str = ""

    def __post_init__(self):
        if self.schema != TAG or not self.task.strip():
            raise ConfigurationError("Flash config needs its schema and one nonempty trained task")
        numbers = (
            self.fps,
            self.chunk_size,
            self.execute_steps,
            self.blend_steps,
            *self.image_size,
        )
        if any(type(v) is not int for v in numbers):
            raise ConfigurationError("Flash rates, horizons and image sizes must be integers")
        if not 20 <= self.fps <= 100 or len(self.image_size) != 2:
            raise ConfigurationError("Flash fps must be 20..100; image_size is [height,width]")
        if not all(32 <= v <= 1024 for v in self.image_size):
            raise ConfigurationError("Flash image dimensions must be 32..1024")
        if not 1 <= self.blend_steps <= self.execute_steps:
            raise ConfigurationError("need 1 <= blend_steps <= execute_steps")
        if not self.execute_steps + self.blend_steps <= self.chunk_size <= 500:
            raise ConfigurationError("chunk must contain executed prefix plus blend overlap")
        if set(self.masks) - set(CAMERAS):
            raise ConfigurationError("unknown camera in operator masks")
        for rectangles in self.masks.values():
            for box in rectangles:
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ConfigurationError("mask rectangle needs four finite coordinates")
                x0, y0, x1, y1 = box
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ConfigurationError("mask rectangle must satisfy 0 <= min < max <= 1")


def prepare_image(frame, camera, config):
    """RGB uint8 HWC or wire JPEG -> ImageNet-normalized float32 CHW."""
    if camera not in CAMERAS:
        raise ConfigurationError(f"unknown camera {camera}")
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise ConfigurationError("Flash images must be uint8 RGB or JPEG bytes")
    if frame.ndim == 1:
        with Image.open(io.BytesIO(frame.tobytes())) as decoded:
            frame = np.asarray(decoded.convert("RGB"))
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ConfigurationError("Flash images must have shape HxWx3")
    pixels = frame.copy()
    height, width = pixels.shape[:2]
    for x0, y0, x1, y1 in config.masks.get(camera, []):
        pixels[
            int(y0 * height) : math.ceil(y1 * height), int(x0 * width) : math.ceil(x1 * width)
        ] = (124, 116, 104)
    height, width = config.image_size
    pixels = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.BILINEAR))
    result = pixels.astype(np.float32).transpose(2, 0, 1) / 255
    return (result - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )[:, None, None]


def flip_grippers(values):
    """Native 1=open <-> dataset 0=open, once per boundary; radians unchanged."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1:] != (14,) or not np.all(np.isfinite(result)):
        raise ConfigurationError("Flash state/action requires 14 finite values")
    if np.any((result[..., [6, 13]] < 0) | (result[..., [6, 13]] > 1)):
        raise ConfigurationError("Flash grippers must be in [0,1]")
    result[..., [6, 13]] = 1 - result[..., [6, 13]]
    return result


def blend_chunks(previous, current, *, executed, overlap):
    """Cubic old-plan decay across aligned chunk overlap, inspired by Flash-ALOHA.

    Zero-valued actions are valid. Overlap is defined by explicit indices. This
    smooths plans; Karma's bounded executor still limits actual position steps.
    """
    result = np.asarray(current, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 14 or not np.isfinite(result).all():
        raise InferenceError("Flash chunk must have shape (N,14), all finite")
    if not 1 <= overlap <= executed or len(result) < executed + overlap:
        raise InferenceError("Flash response is too short for its execution/overlap contract")
    if previous is not None:
        previous = np.asarray(previous)
        if previous.shape != result.shape or not np.isfinite(previous).all():
            raise InferenceError("Flash chunk shape changed during execution")
        # End at the new trajectory exactly, avoiding a residual at the boundary.
        weight = (1 - np.linspace(0, 1, overlap) ** 3)[:, None]
        if overlap == 1:
            weight[:] = 0
        result[:overlap] = (
            weight * previous[executed : executed + overlap] + (1 - weight) * result[:overlap]
        )
    return result


class FlashChunkClient(MolmoActClient):
    """Synchronous chunk client; holds during inference, then executes a prefix.

    Do not wrap in ChunkPrefetcher: overlap indices assume the previous prefix
    has finished. Its bounded executor may slow actions further on large deltas.
    """

    def __init__(self, server=None, *, fps=30, instruction=None, **kwargs):
        super().__init__(server, contract=CONTRACT, enable_cuda_graph=False, **kwargs)
        self.fps = fps
        self.instruction = instruction
        self.config = None
        self.previous = None

    def health(self):
        payload = super().health()
        try:
            self.config = FlashConfig(**payload["flash_config"])
        except (KeyError, TypeError, ValueError) as err:
            raise InferenceError(f"invalid Flash checkpoint contract: {err}") from err
        if payload.get("gripper_open") != 1 or payload.get("cameras") != list(CAMERAS):
            raise InferenceError("Flash camera order or native gripper polarity mismatch")
        if self.config.fps != self.fps:
            raise InferenceError("control rate must equal the Flash checkpoint data rate")
        if self.instruction is not None and self.instruction != self.config.task:
            raise InferenceError("ACT instruction must match the trained task before powering up")
        self.previous = None
        return payload

    def infer(self, observation, instruction, **kwargs):
        if self.config is None:
            raise InferenceError("Flash health must succeed before inference")
        if instruction != self.config.task:
            raise InferenceError("ACT is task-specific; instruction must match the trained task")
        try:
            actions = super().infer(observation, instruction, **kwargs)
            if len(actions) != self.config.chunk_size:
                raise InferenceError("Flash server returned the wrong chunk length")
            actions = blend_chunks(
                self.previous,
                actions,
                executed=self.config.execute_steps,
                overlap=self.config.blend_steps,
            )
            self.previous = actions
            return actions[: self.config.execute_steps].copy()
        except BaseException:
            self.previous = None
            raise
