"""Resize frames on the GPU with FFmpeg's scale_vaapi filter, keeping them in GPU memory.

PyAV 18's filter graph can run scale_vaapi, but it can't give the graph's buffer
source the frames context of the GPU frames it will receive, which hardware
filters need. This module fills that gap: it calls av_buffersrc_parameters_set
through ctypes, on the same libavfilter PyAV loaded. That needs a few raw
pointers PyAV doesn't expose, so their offsets are pinned below for PyAV 18.1 and
FFmpeg 8, and each pointer is checked against a value PyAV does expose first.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import av
from av.filter import Graph
from av.filter.context import FilterContext

POINTER = ctypes.sizeof(ctypes.c_void_p)
# A Cython object is PyObject_HEAD, a vtable pointer if the class has cdef
# methods, then its C fields in declaration order (av/frame.pxd, av/filter/context.pxd).
FRAME_PTR = object.__basicsize__ + POINTER  # Frame.ptr, an AVFrame *
FILTER_CONTEXT_PTR = object.__basicsize__  # FilterContext.ptr, an AVFilterContext *
# offsetof() in the FFmpeg 8.0 headers (libavutil 60, libavfilter 11). These are
# public structs, so they only change with a major version.
AVFRAME_WIDTH, AVFRAME_HEIGHT, AVFRAME_FORMAT = 104, 108, 116
AVFRAME_HW_FRAMES_CTX = 328
AVBUFFERREF_DATA = 8
AVHWFRAMESCONTEXT_FORMAT, AVHWFRAMESCONTEXT_SW_FORMAT = 60, 64
AVFILTERCONTEXT_NAME = 16
BUFFERSRC_PARAMS_FORMAT, BUFFERSRC_PARAMS_HW_FRAMES_CTX = 0, 40

PYAV_VERSION = "18.1."
LIBRARY_MAJORS = {"libavutil": 60, "libavfilter": 11}
VAAPI = int(av.VideoFormat("vaapi"))


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


def scaled_size(request: Size, source: Size) -> Size:
    """The output size for a requested WxH, where -1 on one side keeps the source's aspect ratio."""
    if request.width == -1:
        return Size(_even(request.height * source.width / source.height), request.height)
    if request.height == -1:
        return Size(request.width, _even(request.width * source.height / source.width))
    return request


def _even(x: float) -> int:
    # 4:2:0 video needs even dimensions.
    return max(2, 2 * round(x / 2))


def unsupported_reason() -> str | None:
    """Why GPU scaling can't run with this PyAV build, or None if it can."""
    if not av.__version__.startswith(PYAV_VERSION):
        return f"GPU scaling is written for PyAV {PYAV_VERSION}x, not {av.__version__}"
    for library, major in LIBRARY_MAJORS.items():
        if (version := av.library_versions[library])[0] != major:
            return f"GPU scaling is written for FFmpeg 8, but PyAV uses {library} {version}"
    return None


class GpuScaler:
    """Resizes VA-API frames with scale_vaapi."""

    def __init__(self, size: Size):
        if reason := unsupported_reason():
            raise RuntimeError(reason)
        self.size = size
        self._graph: Graph | None = None
        self._input: tuple[int, int, int] | None = None  # (sw_format, width, height) of the graph

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        frames_ref, sw_format = _frames_context(frame)
        # When the decoder is flushed, it replaces its pool of GPU frames. With
        # B-frames, keyframe mode flushes it for every pick. A graph can take
        # surfaces from any pool on the same device, so it's only rebuilt when
        # the frames' format or size changes.
        if self._graph is None or self._input != (sw_format, frame.width, frame.height):
            self._graph = self._build(frame, frames_ref)
            self._input = (sw_format, frame.width, frame.height)
        self._graph.vpush(frame)
        return self._graph.vpull()

    def _build(self, frame: av.VideoFrame, frames_ref: int) -> Graph:
        graph = Graph()
        # FFmpeg 8 won't initialize a buffer source with a hardware format until it
        # has a frames context, and PyAV can't pass one. So the source starts out
        # as nv12 and is switched to VA-API before the graph is configured.
        # Frames drained from the decoder have no time base, and the worker sets
        # timestamps after scaling, so the graph uses a fixed one.
        source = graph.add_buffer(
            width=frame.width, height=frame.height, format="nv12", time_base=Fraction(1, 90000)
        )
        _set_hw_frames(source, frames_ref)
        scale = graph.add("scale_vaapi", w=str(self.size.width), h=str(self.size.height))
        sink = graph.add("buffersink")
        source.link_to(scale)
        scale.link_to(sink)
        graph.configure()
        return graph


LAYOUT_ERROR = "unexpected PyAV object layout; GPU scaling needs PyAV 18.1 with FFmpeg 8"


def _pointer(address: int) -> int:
    return ctypes.c_void_p.from_address(address).value or 0


def _int(address: int) -> int:
    return ctypes.c_int.from_address(address).value


def _frames_context(frame: av.VideoFrame) -> tuple[int, int]:
    """The frame's hw_frames_ctx reference, and the software format of its surfaces."""
    avframe = _pointer(id(frame) + FRAME_PTR)
    fields = avframe and (
        _int(avframe + AVFRAME_WIDTH),
        _int(avframe + AVFRAME_HEIGHT),
        _int(avframe + AVFRAME_FORMAT),
    )
    if fields != (frame.width, frame.height, VAAPI):
        raise RuntimeError(f"{LAYOUT_ERROR} (or {frame} isn't a VA-API frame)")
    ref = _pointer(avframe + AVFRAME_HW_FRAMES_CTX)
    frames_ctx = ref and _pointer(ref + AVBUFFERREF_DATA)
    if not frames_ctx or _int(frames_ctx + AVHWFRAMESCONTEXT_FORMAT) != VAAPI:
        raise RuntimeError(f"{frame} has no VA-API frames context")
    return ref, _int(frames_ctx + AVHWFRAMESCONTEXT_SW_FORMAT)


def _set_hw_frames(source: FilterContext, frames_ref: int) -> None:
    ctx = _pointer(id(source) + FILTER_CONTEXT_PTR)
    name = ctx and _pointer(ctx + AVFILTERCONTEXT_NAME)
    if not name or ctypes.string_at(name).decode() != source.name:
        raise RuntimeError(LAYOUT_ERROR)
    alloc, set_parameters, free = _buffersrc_api()
    params = alloc()
    if not params:
        raise MemoryError("av_buffersrc_parameters_alloc() failed")
    try:
        ctypes.c_int.from_address(params + BUFFERSRC_PARAMS_FORMAT).value = VAAPI
        # Borrowed: av_buffersrc_parameters_set() takes its own reference.
        ctypes.c_void_p.from_address(params + BUFFERSRC_PARAMS_HW_FRAMES_CTX).value = frames_ref
        error = set_parameters(ctx, params)
    finally:
        free(params)
    if error < 0:
        raise RuntimeError(f"av_buffersrc_parameters_set() failed with error {error}")


@cache
def _buffersrc_api():
    libavfilter, libavutil = _loaded_library("libavfilter"), _loaded_library("libavutil")
    alloc = libavfilter.av_buffersrc_parameters_alloc
    alloc.argtypes, alloc.restype = [], ctypes.c_void_p
    set_parameters = libavfilter.av_buffersrc_parameters_set
    set_parameters.argtypes, set_parameters.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
    free = libavutil.av_free
    free.argtypes, free.restype = [ctypes.c_void_p], None
    return alloc, set_parameters, free


def _loaded_library(prefix: str) -> ctypes.CDLL:
    """The copy of an FFmpeg library PyAV loaded; PyPI wheels bundle theirs under another name."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and Path(path := fields[5].strip()).name.startswith(prefix):
                return ctypes.CDLL(path)
    raise RuntimeError(f"{prefix} isn't loaded")
