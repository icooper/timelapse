import math
from fractions import Fraction

import pytest

from timelapse.errors import TimelapseError
from timelapse.sampling import (
    KeyframeStats,
    keyframes_suffice,
    plan_chunks,
    prefer_earlier,
    sample_interval,
    sample_times,
)


@pytest.mark.parametrize(
    "speedup,fps,step",
    [
        (300, 25, 12),
        (300, 20, 15),
        (60, 20, 3),
        (100, 30, Fraction(10, 3)),
        (300, Fraction(30000, 1001), Fraction(1001, 100)),
    ],
)
def test_sample_interval_is_speedup_over_fps(speedup, fps, step):
    assert sample_interval(speedup, fps) == step


def test_sample_times_cover_range_without_trailing_point():
    assert sample_times(0, 60, Fraction(15)) == [0, 15, 30, 45]
    assert sample_times(0, 60.5, Fraction(15)) == [0, 15, 30, 45, 60]
    assert sample_times(10, 40, Fraction(15)) == [10, 25]
    assert len(sample_times(0, 13020.68, Fraction(15))) == 869


def test_sample_times_stay_exact_for_fractional_steps():
    times = sample_times(0, 1000, Fraction(10, 3))
    assert times[3] == 10.0
    assert times[299] == pytest.approx(996.6666667)
    assert len(times) == 300


def test_sample_times_rejects_short_range():
    with pytest.raises(TimelapseError):
        sample_times(5, 5, Fraction(12))


@pytest.mark.parametrize("count,workers", [(869, 4), (100, 3), (7, 2), (3, 8), (1, 1)])
def test_plan_chunks_is_contiguous_and_balanced(count, workers):
    times = [float(i) for i in range(count)]
    chunks = plan_chunks(times, workers)
    assert [t for c in chunks for t in c.targets] == times
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == min(count, workers)
    assert max(c.frames for c in chunks) - min(c.frames for c in chunks) <= 1


# Camera footage: GOP of 20 frames → keyframes ~1.0s apart, up to 1.2s when it drops to ~17 fps.
CAMERA = KeyframeStats(typical=1.002, worst=1.204)


@pytest.mark.parametrize(
    "step,expected",
    [
        (15, True),  # keyframes small next to the step
        (12.5, True),  # not a whole number of intervals, but still ≥ 4x the worst gap
        (3, True),  # whole number of keyframe intervals
        (1, True),  # every keyframe
        (2.4, False),  # samples would alternate between 2 and 3 keyframes apart
        (0.5, False),  # shorter than a keyframe interval
    ],
)
def test_keyframes_suffice(step, expected):
    assert keyframes_suffice(step, CAMERA) is expected


def test_keyframes_never_suffice_when_unmeasurable():
    assert not keyframes_suffice(600, KeyframeStats(math.inf, math.inf))


@pytest.mark.parametrize(
    "target,before,after,expected",
    [
        (10.0, 9.5, 10.7, True),
        (10.0, 9.0, 10.5, False),
        (10.0, 9.5, 10.5, True),  # ties go to the earlier frame
        (10.0, None, 10.5, False),
        (10.0, 9.0, None, True),
    ],
)
def test_prefer_earlier(target, before, after, expected):
    assert prefer_earlier(target, before, after) is expected
