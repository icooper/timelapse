"""End-to-end runs on a real GPU. Skipped unless VA-API encoding works (i.e. inside the container)."""

from fractions import Fraction

import av
import pytest
from videos import decode_indices, make_video

from timelapse.pipeline import Settings, make_plan, run_plan

pytestmark = pytest.mark.gpu
FPS = 20


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    folder = tmp_path_factory.mktemp("videos")
    return {
        "gop20": make_video(folder / "gop20.mp4", seconds=60, fps=FPS, gop=20),
        "gop35": make_video(folder / "gop35.mp4", seconds=60, fps=FPS, gop=35),
    }


def render(source, tmp_path, device, **settings):
    output = tmp_path / f"out_{len(list(tmp_path.iterdir()))}.mp4"
    plan = make_plan(source, output, Settings(fps=Fraction(FPS), **settings), [device])
    results = run_plan(plan)
    return plan, results, output


def nearest_multiple(frame: int, gop: int) -> int:
    below = frame // gop * gop
    return below if frame - below <= below + gop - frame else below + gop


def test_keyframe_mode_on_aligned_grid_picks_grid_frames(videos, tmp_path, gpu_device):
    # 100x at 20 fps → 5s step = 100 frames, a whole number of 1s keyframe intervals.
    plan, results, output = render(videos["gop20"], tmp_path, gpu_device, speedup=100)
    assert plan.keyframes_only
    assert decode_indices(output) == [100 * k for k in range(12)]
    assert sum(r.decoded for r in results) == 12  # one decode per output frame


def test_keyframe_mode_picks_nearest_keyframe(videos, tmp_path, gpu_device):
    plan, _, output = render(videos["gop35"], tmp_path, gpu_device, speedup=100, mode="keyframes")
    assert decode_indices(output) == [nearest_multiple(100 * k, 35) for k in range(12)]


def test_exact_mode_picks_exact_frames(videos, tmp_path, gpu_device):
    plan, results, output = render(videos["gop35"], tmp_path, gpu_device, speedup=100, mode="exact")
    assert not plan.keyframes_only
    assert decode_indices(output) == [100 * k for k in range(12)]
    assert sum(r.decoded for r in results) < 12 * 35  # decodes from the keyframe before, not everything


def test_exact_mode_with_small_steps_decodes_forward(videos, tmp_path, gpu_device):
    # 30x at 20 fps → 1.5s step: shorter than the 1.75s keyframe interval, so no re-seeking.
    _, results, output = render(videos["gop35"], tmp_path, gpu_device, speedup=30, mode="exact", end=30)
    assert decode_indices(output) == [30 * k for k in range(20)]
    assert sum(r.decoded for r in results) <= 30 * 19 + 35


def test_chunks_join_seamlessly(videos, tmp_path, gpu_device):
    _, _, single = render(videos["gop35"], tmp_path, gpu_device, speedup=100, mode="exact", jobs_per_gpu=1)
    plan, _, chunked = render(
        videos["gop35"], tmp_path, gpu_device, speedup=100, mode="exact", jobs_per_gpu=3
    )
    assert len(plan.chunks) == 3
    assert decode_indices(chunked) == decode_indices(single)


def test_output_is_tagged_hevc_at_output_rate(videos, tmp_path, gpu_device):
    _, _, output = render(videos["gop20"], tmp_path, gpu_device, speedup=100, jobs_per_gpu=2)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "hevc"
        assert stream.codec_tag == "hvc1"
        assert stream.average_rate == FPS
        assert stream.frames == 12
        assert float(container.duration / av.time_base) == pytest.approx(12 / FPS)


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_other_codecs(videos, tmp_path, gpu_device, codec):
    _, _, output = render(videos["gop20"], tmp_path, gpu_device, speedup=100, codec=codec)
    assert decode_indices(output, gpu_device) == [100 * k for k in range(12)]
