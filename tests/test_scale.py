from fractions import Fraction

import av
import pytest
from av.filter import Graph

from timelapse import scale
from timelapse.scale import Size, scaled_size, unsupported_reason


@pytest.mark.parametrize(
    "request_,size",
    [
        (Size(1920, 1080), Size(1920, 1080)),
        (Size(1920, -1), Size(1920, 1080)),
        (Size(-1, 720), Size(1280, 720)),
        (Size(1000, -1), Size(1000, 562)),  # 562.5 rounds to an even height
    ],
)
def test_scaled_size(request_, size):
    assert scaled_size(request_, Size(3840, 2160)) == size


def test_gpu_scaling_supports_the_pinned_pyav():
    assert unsupported_reason() is None


def test_gpu_scaling_refuses_other_pyav_versions(monkeypatch):
    monkeypatch.setattr(av, "__version__", "19.0.0")
    assert "19.0.0" in unsupported_reason()


def test_pinned_offsets_match_pyav_objects():
    # The ctypes bridge reads PyAV's C pointers at fixed offsets; check them on objects we can inspect.
    frame = av.VideoFrame(64, 48, "nv12")
    avframe = scale._pointer(id(frame) + scale.FRAME_PTR)
    assert (scale._int(avframe + scale.AVFRAME_WIDTH), scale._int(avframe + scale.AVFRAME_HEIGHT)) == (64, 48)
    with pytest.raises(RuntimeError, match="VA-API"):
        scale._frames_context(frame)

    graph = Graph()  # a filter context doesn't keep its graph alive
    source = graph.add_buffer(width=64, height=48, format="nv12", time_base=Fraction(1, 20))
    ctx = scale._pointer(id(source) + scale.FILTER_CONTEXT_PTR)
    name = scale._pointer(ctx + scale.AVFILTERCONTEXT_NAME)
    assert scale.ctypes.string_at(name).decode() == source.name


def test_bridge_uses_the_libraries_pyav_loaded():
    alloc, _, free = scale._buffersrc_api()
    params = alloc()
    assert params
    free(params)
