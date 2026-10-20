"""End-to-end runs on a real GPU. Skipped unless VA-API encoding works (i.e. inside the container)."""

from fractions import Fraction

import av
import pytest
from videos import decode_indices, make_video, read_index

from timelapse.pipeline import Settings, make_plan, run_plan
from timelapse.scale import AVBUFFERREF_DATA, GpuScaler, Size, _frames_context, _pointer
from timelapse.worker import Decoder, FramePicker

pytestmark = pytest.mark.gpu
FPS = 20


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    folder = tmp_path_factory.mktemp("videos")
    return {
        "gop20": make_video(folder / "gop20.mp4", seconds=60, fps=FPS, gop=20),
        "gop35": make_video(folder / "gop35.mp4", seconds=60, fps=FPS, gop=35),
        "gop35_ipp": make_video(folder / "gop35_ipp.mp4", seconds=60, fps=FPS, gop=35, bframes=False),
    }


@pytest.fixture
def open_picker(gpu_device):
    containers = []

    def open_picker(path, **kwargs):
        container = av.open(str(path))
        containers.append(container)
        stream = container.streams.video[0]
        return FramePicker(container, stream, Decoder(stream, gpu_device), **kwargs)

    yield open_picker
    for container in containers:
        container.close()


def frames_context(frame) -> int:
    """Address of the frame's AVHWFramesContext: the same for every frame from one pool."""
    return _pointer(_frames_context(frame)[0] + AVBUFFERREF_DATA)


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


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("keyframes", [nearest_multiple(100 * k, 35) for k in range(12)]),
        ("exact", [100 * k for k in range(12)]),
    ],
)
def test_video_without_b_frames(videos, tmp_path, gpu_device, mode, expected):
    _, _, output = render(videos["gop35_ipp"], tmp_path, gpu_device, speedup=100, mode=mode)
    assert decode_indices(output) == expected


def test_decoder_keeps_its_pool_across_seeks(videos, open_picker):
    # On FFmpeg 8.0, flushing the decoder after each seek made it build a new pool
    # of GPU frames, at about 4.5 ms per 4K seek. Without B-frames, it never needs a flush.
    picker = open_picker(videos["gop35_ipp"], keyframes_only=True, seek_ahead=2)
    times = (5, 20, 40, 55)
    frames = [picker.pick(t) for t in times]  # kept alive so pool addresses aren't reused
    assert [read_index(f) for f in frames] == [nearest_multiple(t * FPS, 35) for t in times]
    assert len({frames_context(f) for f in frames}) == 1
    assert picker.decoder.flushes == 0


def test_exact_mode_reseeks_backward_without_stale_frames(videos, open_picker):
    # Between frames, the next target can fall before the frame read ahead, so
    # the picker seeks back. With B-frames, the decoder still holds later frames
    # then, and it must flush rather than return one of them.
    picker = open_picker(videos["gop35"], keyframes_only=False, seek_ahead=2)
    assert [read_index(picker.pick(t)) for t in (5.0, 5.01, 5.02, 30.0)] == [100, 100, 100, 600]
    assert picker.decoder.flushes == 1


def test_decoder_drops_frames_held_from_before_a_seek(videos, gpu_device):
    with av.open(str(videos["gop35"])) as container:
        stream = container.streams.video[0]
        decoder = Decoder(stream, gpu_device)
        packets = container.demux(stream)
        for _ in range(10):  # leaves frames held back for reordering
            decoder.decode(next(packets))
        container.seek(int(20 / stream.time_base), stream=stream)
        decoder.seek()
        packets = container.demux(stream)
        indices = [read_index(f) for _ in range(10) for f in decoder.decode(next(packets))]
    assert indices and min(indices) == 385  # the keyframe at or before 20 s
    assert decoder.flushes == 0


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


@pytest.mark.parametrize(
    "video,mode,expected",
    [
        ("gop20", "keyframes", [100 * k for k in range(12)]),
        ("gop35", "keyframes", [nearest_multiple(100 * k, 35) for k in range(12)]),
        ("gop35", "exact", [100 * k for k in range(12)]),
    ],
)
def test_scale_resizes_on_the_gpu(videos, tmp_path, gpu_device, video, mode, expected):
    plan, _, output = render(
        videos[video], tmp_path, gpu_device, speedup=100, mode=mode, scale=Size(320, -1), jobs_per_gpu=2
    )
    assert plan.size == Size(320, 180)
    with av.open(str(output)) as container:
        context = container.streams.video[0].codec_context
        assert (context.width, context.height) == (320, 180)
    assert decode_indices(output) == expected


def test_scaler_keeps_its_graph_across_pools(videos, open_picker):
    # FFmpeg 8.0 made a new pool of GPU frames on every flush, which keyframe mode
    # does for every pick with B-frames; rebuilding the filter graph for each pool
    # cost about 4 ms per frame. FFmpeg 8.1 keeps the pool, so frames from a second
    # decoder stand in for a new one.
    scaler = GpuScaler(Size(320, 180))
    first = open_picker(videos["gop20"], keyframes_only=True, seek_ahead=2).pick(5)
    scaler(first)
    graph = scaler._graph
    picker = open_picker(videos["gop20"], keyframes_only=True, seek_ahead=2)
    for t in (20, 40, 50):
        source = picker.pick(t)
        assert frames_context(source) != frames_context(first)
        frame = scaler(source)
        assert (frame.width, frame.height) == (320, 180)
        assert scaler._graph is graph


def test_scale_can_stretch(videos, tmp_path, gpu_device):
    _, _, output = render(videos["gop20"], tmp_path, gpu_device, speedup=100, scale=Size(480, 480))
    with av.open(str(output)) as container:
        context = container.streams.video[0].codec_context
        assert (context.width, context.height) == (480, 480)
    assert decode_indices(output) == [100 * k for k in range(12)]


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_other_codecs(videos, tmp_path, gpu_device, codec):
    _, _, output = render(videos["gop20"], tmp_path, gpu_device, speedup=100, codec=codec)
    assert decode_indices(output, gpu_device) == [100 * k for k in range(12)]


def test_av1_input(videos, tmp_path, gpu_device):
    # Every other frame of gop20, with a keyframe every 2 s: too far apart for
    # keyframe mode, so this decodes forward from them.
    _, _, source = render(videos["gop20"], tmp_path, gpu_device, speedup=2, codec="av1")
    plan, _, output = render(source, tmp_path, gpu_device, speedup=10)
    assert not plan.keyframes_only
    assert decode_indices(output) == [20 * k for k in range(60)]


@pytest.mark.parametrize("suffix,format_name", [(".mov", "mov"), (".mkv", "matroska")])
def test_other_containers(videos, tmp_path, gpu_device, suffix, format_name):
    output = tmp_path / f"out{suffix}"
    run_plan(make_plan(videos["gop20"], output, Settings(fps=Fraction(FPS), speedup=100), [gpu_device]))
    with av.open(str(output)) as container:
        assert format_name in container.format.name
    assert decode_indices(output) == [100 * k for k in range(12)]
