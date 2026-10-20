"""Render one chunk: seek to each sample, decode on the GPU, encode on the GPU, write a segment.

This runs in a worker process. Decoded frames stay in GPU memory, and so do
frames resized by scale_vaapi: the encoder adopts the frames context of the
first frame it gets, so everything shares the decoder's VA-API connection and no
frame is copied to system memory.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
from av.codec.context import CodecContext
from av.codec.hwaccel import HWAccel
from av.video.frame import PictureType

from timelapse.gpu import ENCODERS
from timelapse.sampling import Chunk, prefer_earlier
from timelapse.scale import GpuScaler, Size


class ProgressSink(Protocol):
    def put(self, item: tuple[int, int]) -> None: ...


@dataclass(frozen=True)
class ChunkJob:
    chunk: Chunk
    device: str
    source: Path
    segment: Path
    fps: Fraction
    size: Size  # output size; frames are resized on the GPU if it differs from the source's
    codec: str
    quality: int
    keyframes_only: bool
    # In exact mode, decode forward instead of seeking when the next sample is
    # at most this many seconds past the last decoded frame.
    seek_ahead: float


@dataclass(frozen=True)
class ChunkResult:
    index: int
    frames: int
    decoded: int
    seconds: float


class Decoder:
    """Decodes a stream on the GPU, without the reinitialization a flush causes.

    container.seek() flushes the stream's own decoder, and after a flush FFmpeg
    8.0's H.264 decoder rebuilt its hwaccel, with a new pool of GPU frames. At 4K
    that cost about 4.5 ms per seek. (FFmpeg 8.1 keeps the pool.) Seeks don't
    touch this decoder, so it's flushed only when it has to be: after draining,
    or before going back to a packet it has already decoded.
    """

    def __init__(self, stream, device: str):
        hwaccel = HWAccel("vaapi", device, allow_software_fallback=False, is_hw_owned=True)
        self.context = CodecContext.create(stream.codec_context.codec, "r", hwaccel=hwaccel)
        # Unlike av.open(), CodecContext.create() falls back to software silently.
        if not self.context.is_hwaccel:
            raise RuntimeError(f"{stream.codec_context.name} can't be decoded on {device}")
        self.context.extradata = stream.codec_context.extradata
        # PyAV presets pix_fmt to "vaapi". FFmpeg's AV1 decoder takes that to mean
        # its hwaccel is already set up, skips get_format() and can't allocate
        # frames. A software format makes it negotiate the hwaccel.
        if sw_format := stream.codec_context.pix_fmt:
            self.context.pix_fmt = sw_format
        self.flushes = 0
        self._start: int | None = None  # pts of the first packet since the last seek
        self._last: int | None = None  # highest pts decoded since the last flush
        self._drained = False

    def seek(self) -> None:
        """Call after seeking the container: the next packet starts a new run of frames."""
        self._start = None

    def decode(self, packet: av.Packet | None) -> list[av.VideoFrame]:
        """Frames from `packet`; None, or the empty packet that ends a demux, drains the decoder."""
        if packet is None or packet.pts is None:
            self._drained = True
        else:
            if self._start is None:
                if self._drained or (self._last is not None and packet.pts <= self._last):
                    self.context.flush_buffers()
                    self.flushes += 1
                    self._drained, self._last = False, None
                self._start = packet.pts
            self._last = packet.pts if self._last is None else max(self._last, packet.pts)
        frames = self.context.decode(packet)
        # Frames still held for reordering from before a seek. Seeks that don't
        # flush go forward, so all of them come before the new run.
        return [f for f in frames if self._start is None or f.pts >= self._start]


class FramePicker:
    """Returns the frame nearest each requested source time; times must increase."""

    def __init__(self, container, stream, decoder: Decoder, *, keyframes_only: bool, seek_ahead: float):
        self.container = container
        self.stream = stream
        self.decoder = decoder
        self.keyframes_only = keyframes_only
        self.seek_ahead = seek_ahead
        self.decoded = 0
        self._frames: Iterator[av.VideoFrame] | None = None
        self._last: av.VideoFrame | None = None  # last frame pulled from _frames

    def pick(self, t: float) -> av.VideoFrame:
        return self._nearest_keyframe(t) if self.keyframes_only else self._nearest_frame(t)

    def _seek(self, t: float) -> None:
        self.container.seek(max(0, int(t / self.stream.time_base)), stream=self.stream, backward=True)
        self.decoder.seek()

    def _time(self, pts: int) -> float:
        return float(pts * self.stream.time_base)

    def _nearest_keyframe(self, t: float) -> av.VideoFrame:
        # The seek lands on the keyframe at or before t. Demux (without decoding)
        # up to the next keyframe, then decode only whichever of the two is closer.
        self._seek(t)
        before = after = None
        for packet in self.container.demux(self.stream):
            if packet.pts is None:  # end of stream
                break
            if not packet.is_keyframe:
                continue
            if self._time(packet.pts) <= t:
                before = packet
            else:
                after = packet
                break
        chosen = (
            before
            if prefer_earlier(t, before and self._time(before.pts), after and self._time(after.pts))
            else after
        )
        if chosen is None:
            raise RuntimeError(f"no keyframe found near {t:.3f}s")
        # Streams with B-frames hold the keyframe back for reordering. Draining
        # returns it, at the cost of a flush before the next pick.
        frames = self.decoder.decode(chosen) or self.decoder.decode(None)
        self.decoded += 1
        return frames[0]

    def _decode_from(self, t: float) -> Iterator[av.VideoFrame]:
        self._seek(t)
        for packet in self.container.demux(self.stream):  # the final empty packet drains the decoder
            for frame in self.decoder.decode(packet):
                self.decoded += 1
                yield frame

    def _nearest_frame(self, t: float) -> av.VideoFrame:
        last = self._last
        if self._frames is None or last is None or t < last.time or t - last.time > self.seek_ahead:
            self._frames, self._last = self._decode_from(t), None
        before = self._last
        if before is not None and before.time >= t:
            return before  # sampling faster than the source frame rate repeats frames
        for frame in self._frames:
            if frame.time >= t:
                self._last = frame
                return before if prefer_earlier(t, before and before.time, frame.time) else frame
            before = frame
        self._frames = None
        if before is None:
            raise RuntimeError(f"no frame found near {t:.3f}s")
        return before


def render_chunk(job: ChunkJob, progress: ProgressSink | None = None) -> ChunkResult:
    started = time.perf_counter()
    with (
        av.open(str(job.source)) as source,
        av.open(str(job.segment), "w", format="mp4") as segment,
    ):
        stream = source.streams.video[0]
        encoder = segment.add_stream(
            ENCODERS[job.codec],
            rate=job.fps,
            hwaccel=HWAccel("vaapi", job.device),
            options={"rc_mode": "CQP", "global_quality": str(job.quality)},
        )
        source_size = Size(stream.codec_context.width, stream.codec_context.height)
        scaler = GpuScaler(job.size) if job.size != source_size else None
        encoder.width, encoder.height = job.size.width, job.size.height
        # The input frames are GPU surfaces. With pix_fmt "vaapi" the encoder
        # adopts their frames context instead of uploading from system memory.
        encoder.pix_fmt = "vaapi"
        encoder.codec_context.gop_size = max(1, round(2 * job.fps))
        frame_time = 1 / job.fps
        encoder.time_base = frame_time

        written, last_dts = 0, None

        def write(packets: list[av.Packet]) -> None:
            nonlocal written, last_dts
            for packet in packets:
                pts, dts = packet.pts, packet.dts
                if pts is None or dts is None or pts < dts or (last_dts is not None and dts <= last_dts):
                    raise RuntimeError(f"encoder produced invalid timestamps (pts={pts}, dts={dts})")
                last_dts = dts
                segment.mux(packet)
                written += 1

        picker = FramePicker(
            source,
            stream,
            Decoder(stream, job.device),
            keyframes_only=job.keyframes_only,
            seek_ahead=job.seek_ahead,
        )
        for n, t in enumerate(job.chunk.targets):
            frame = picker.pick(t)
            if scaler is not None:
                frame = scaler(frame)
            frame.pts, frame.time_base, frame.duration = n, frame_time, 1
            # Decoded keyframes keep pict_type I, which encoders treat as a forced
            # keyframe; left alone, every output frame would be an IDR frame.
            frame.pict_type = PictureType.NONE
            write(encoder.encode(frame))
            if progress is not None:
                progress.put((job.chunk.index, n + 1))
        write(encoder.encode(None))
        if written != job.chunk.frames:
            raise RuntimeError(f"encoded {written} frames, expected {job.chunk.frames}")

    return ChunkResult(job.chunk.index, job.chunk.frames, picker.decoded, time.perf_counter() - started)
