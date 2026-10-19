"""Sample grid, chunk planning and decode-mode choice. Pure functions, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from timelapse.errors import TimelapseError

# Keyframe-only sampling is used in two cases. One is when keyframes are at most
# this fraction of a step apart; the nearest keyframe is then never more than
# step/8 away from its grid point.
MAX_KEYFRAME_GAP_RATIO = 0.25
# The other is when the step is within this many keyframe intervals of a whole
# number of them, so samples keep landing on keyframes.
KEYFRAME_ALIGN_TOLERANCE = 0.1


@dataclass(frozen=True)
class KeyframeStats:
    typical: float  # median gap between keyframes, in seconds
    worst: float  # largest gap; infinite if a probe window had fewer than two keyframes


@dataclass(frozen=True)
class Chunk:
    index: int
    targets: tuple[float, ...]  # source times to sample, in seconds

    @property
    def frames(self) -> int:
        return len(self.targets)


def sample_interval(speedup: int, fps: Fraction) -> Fraction:
    """Seconds of source per output frame, kept exact so the sample grid doesn't drift."""
    return speedup / Fraction(fps)


def sample_times(start: float, end: float, step: Fraction) -> list[float]:
    """Grid points start + k*step that fall before end."""
    count = math.ceil((end - start) / step - 1e-9)
    if count <= 0:
        raise TimelapseError(f"range {start:g}-{end:g}s is shorter than one step ({float(step):g}s)")
    return [start + float(k * step) for k in range(count)]


def plan_chunks(times: list[float], workers: int) -> list[Chunk]:
    """Split the sample times into contiguous chunks of near-equal size."""
    workers = max(1, min(workers, len(times)))
    chunks, first = [], 0
    for i in range(workers):
        size = len(times) // workers + (1 if i < len(times) % workers else 0)
        chunks.append(Chunk(i, tuple(times[first : first + size])))
        first += size
    return chunks


def keyframes_suffice(step: float, kf: KeyframeStats) -> bool:
    """Whether sampling only keyframes keeps the output timing even.

    Each sample is the keyframe nearest its grid point. That's fine when
    keyframes are close together compared with the step. It's also fine when
    the step is a whole number of keyframe intervals, so samples keep landing
    on keyframes.
    """
    if not math.isfinite(kf.worst):
        return False
    if kf.worst <= step * MAX_KEYFRAME_GAP_RATIO:
        return True
    intervals = step / kf.typical
    return round(intervals) >= 1 and abs(intervals - round(intervals)) <= KEYFRAME_ALIGN_TOLERANCE


def prefer_earlier(target: float, before: float | None, after: float | None) -> bool:
    """Whether the candidate at `before` is at least as close to `target` as `after`."""
    if before is None:
        return False
    if after is None:
        return True
    return target - before <= after - target
