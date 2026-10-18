"""Read video properties and keyframe spacing with PyAV (demuxing only, no decoding)."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av

from timelapse.errors import TimelapseError
from timelapse.sampling import KeyframeStats


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    start: float  # timestamp of the first frame, in seconds
    width: int
    height: int
    codec: str
    frame_rate: Fraction  # nominal rate, used as the default output fps
    avg_fps: float


def probe(path: Path) -> VideoInfo:
    try:
        container = av.open(str(path))
    except av.FFmpegError as e:
        raise TimelapseError(f"can't open {path}: {e}") from e
    with container:
        if not container.streams.video:
            raise TimelapseError(f"{path} has no video stream")
        stream = container.streams.video[0]
        if container.duration:
            duration = container.duration / av.time_base
        elif stream.duration:
            duration = float(stream.duration * stream.time_base)
        else:
            raise TimelapseError(f"{path} has no duration")
        avg = stream.average_rate or Fraction(0)
        nominal = stream.base_rate or Fraction(0)
        # On some VFR files the base rate is a timebase-like value rather than a
        # frame rate. Fall back to the rounded average there.
        if not 0 < nominal <= 240:
            nominal = Fraction(round(avg)) if avg > 0 else Fraction(25)
        return VideoInfo(
            duration=duration,
            start=float(stream.start_time * stream.time_base) if stream.start_time else 0.0,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=stream.codec_context.name,
            frame_rate=nominal,
            avg_fps=float(avg),
        )


def keyframe_stats(path: Path, info: VideoInfo, windows: int = 8, length: float = 30.0) -> KeyframeStats:
    """Measure keyframe spacing across evenly spaced windows of the file."""
    length = min(length, info.duration)
    span = info.duration - length
    starts = [info.start + span * i / (windows - 1) for i in range(windows)] if span > 0 else [info.start]
    gaps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for start in starts:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
            keyframes = []
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                t = float(packet.pts * stream.time_base)
                if t > start + length:
                    break
                if packet.is_keyframe:
                    keyframes.append(t)
            keyframes.sort()
            if len(keyframes) < 2:
                return KeyframeStats(math.inf, math.inf)
            gaps += [b - a for a, b in pairwise(keyframes)]
    return KeyframeStats(statistics.median(gaps), max(gaps))
