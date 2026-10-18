"""Synthetic test videos whose frames carry their own index, so tests can tell which frame was picked."""

from __future__ import annotations

from pathlib import Path

import av
import av.codec.hwaccel
import numpy as np

WIDTH, HEIGHT = 640, 360
BASE, SPACING = 50, 4  # index = hi * BASE + lo, each painted as a flat gray level 16 + SPACING * digit


def paint(index: int) -> np.ndarray:
    """A yuv420p frame: the left half's luma encodes index % BASE, the right half index // BASE."""
    data = np.full((HEIGHT * 3 // 2, WIDTH), 128, np.uint8)
    data[:HEIGHT, : WIDTH // 2] = 16 + SPACING * (index % BASE)
    data[:HEIGHT, WIDTH // 2 :] = 16 + SPACING * (index // BASE)
    return data


def read_index(frame: av.VideoFrame) -> int:
    luma = frame.reformat(format="yuv420p").to_ndarray()[: frame.height]
    w = frame.width
    lo = round((float(luma[:, w // 8 : 3 * w // 8].mean()) - 16) / SPACING)
    hi = round((float(luma[:, 5 * w // 8 : 7 * w // 8].mean()) - 16) / SPACING)
    return hi * BASE + lo


def make_video(path: Path, *, seconds: float, fps: int = 20, gop: int = 20) -> Path:
    """H.264 with a keyframe exactly every `gop` frames (and B-frames, like camera footage)."""
    with av.open(str(path), "w") as out:
        stream = out.add_stream(
            "libx264",
            rate=fps,
            options={"preset": "veryfast", "x264-params": f"keyint={gop}:min-keyint={gop}:scenecut=0"},
        )
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(round(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(paint(i), format="yuv420p")
            frame.pts = i
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def decode_indices(path: Path, device: str | None = None) -> list[int]:
    """Indices of every frame in `path`, decoding on `device` if given (needed for AV1 here)."""
    hwaccel = av.codec.hwaccel.HWAccel("vaapi", device) if device else None
    with av.open(str(path), hwaccel=hwaccel) as container:
        return [read_index(frame) for frame in container.decode(video=0)]
