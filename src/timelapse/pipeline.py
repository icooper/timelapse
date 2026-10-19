"""Plan a time-lapse and run its chunks in parallel across the GPUs."""

from __future__ import annotations

import math
import queue
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing import Manager
from pathlib import Path

from timelapse.concat import concat_segments
from timelapse.errors import TimelapseError
from timelapse.probe import VideoInfo, keyframe_stats, probe
from timelapse.sampling import (
    Chunk,
    KeyframeStats,
    keyframes_suffice,
    plan_chunks,
    sample_interval,
    sample_times,
)
from timelapse.scale import Size, scaled_size
from timelapse.worker import ChunkJob, ChunkResult, render_chunk

# CQP quality per codec. av1_vaapi uses the 0-255 q_idx scale; 90 lands close
# to HEVC at 22 in size.
DEFAULT_QUALITY = {"hevc": 22, "h264": 20, "av1": 90}
MODES = ("auto", "keyframes", "exact")


@dataclass(frozen=True)
class Settings:
    speedup: int = 300
    fps: Fraction | None = None  # None means the input's frame rate
    scale: Size | None = None  # None keeps the input's size; -1 on one side keeps its aspect ratio
    codec: str = "hevc"
    quality: int | None = None
    jobs_per_gpu: int = 2
    mode: str = "auto"
    start: float = 0.0
    end: float | None = None


@dataclass(frozen=True)
class Plan:
    source: Path
    output: Path
    info: VideoInfo
    keyframes: KeyframeStats
    fps: Fraction
    size: Size
    step: Fraction
    keyframes_only: bool
    codec: str
    quality: int
    chunks: list[Chunk]
    devices: list[str]

    @property
    def frames(self) -> int:
        return sum(c.frames for c in self.chunks)

    def device_for(self, chunk: Chunk) -> str:
        return self.devices[chunk.index % len(self.devices)]


def make_plan(source: Path, output: Path, settings: Settings, devices: list[str]) -> Plan:
    if settings.mode not in MODES:
        raise TimelapseError(f"unknown mode {settings.mode!r}")
    if not devices:
        raise TimelapseError("no GPU devices to run on")
    info = probe(source)
    end = info.duration if settings.end is None else min(settings.end, info.duration)
    if not 0 <= settings.start < end:
        raise TimelapseError(f"invalid range {settings.start:g}-{end:g}s (duration {info.duration:.1f}s)")
    fps = settings.fps or info.frame_rate
    source_size = Size(info.width, info.height)
    size = scaled_size(settings.scale, source_size) if settings.scale else source_size
    step = sample_interval(settings.speedup, fps)
    kf = keyframe_stats(source, info)
    if settings.mode == "auto":
        keyframes_only = keyframes_suffice(float(step), kf)
    else:
        keyframes_only = settings.mode == "keyframes"
    times = sample_times(info.start + settings.start, info.start + end, step)
    return Plan(
        source=source,
        output=output,
        info=info,
        keyframes=kf,
        fps=fps,
        size=size,
        step=step,
        keyframes_only=keyframes_only,
        codec=settings.codec,
        quality=settings.quality if settings.quality is not None else DEFAULT_QUALITY[settings.codec],
        chunks=plan_chunks(times, len(devices) * settings.jobs_per_gpu),
        devices=devices,
    )


def seek_ahead(kf: KeyframeStats) -> float:
    """How far ahead exact mode decodes forward before seeking is cheaper."""
    return 2 * kf.typical if math.isfinite(kf.typical) else 10.0


def run_plan(
    plan: Plan,
    *,
    keep_temp: bool = False,
    on_progress: Callable[[dict[int, int]], None] | None = None,
) -> list[ChunkResult]:
    output = plan.output.resolve()
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".timelapse-", delete=not keep_temp) as tmp:
        jobs = [
            ChunkJob(
                chunk=chunk,
                device=plan.device_for(chunk),
                source=plan.source.resolve(),
                segment=Path(tmp) / f"seg_{chunk.index:03d}.mp4",
                fps=plan.fps,
                size=plan.size,
                codec=plan.codec,
                quality=plan.quality,
                keyframes_only=plan.keyframes_only,
                seek_ahead=seek_ahead(plan.keyframes),
            )
            for chunk in plan.chunks
        ]
        results = run_jobs(jobs, on_progress)
        frames = concat_segments([job.segment for job in jobs], output, plan.fps, plan.codec)
        if frames != plan.frames:
            raise TimelapseError(f"expected {plan.frames} frames but wrote {frames}")
    return results


def run_jobs(
    jobs: list[ChunkJob], on_progress: Callable[[dict[int, int]], None] | None = None
) -> list[ChunkResult]:
    """Render chunks in worker processes, reporting frames done per chunk."""
    done = {job.chunk.index: 0 for job in jobs}

    def drain(updates, timeout: float) -> None:
        try:
            while True:
                index, count = updates.get(timeout=timeout)
                done[index] = count
                timeout = 0
        except queue.Empty:
            pass

    with Manager() as manager, ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        updates = manager.Queue()
        futures = [pool.submit(render_chunk, job, updates) for job in jobs]
        while not all(f.done() for f in futures):
            drain(updates, timeout=0.1)
            if on_progress:
                on_progress(done)
            failed = next((f for f in futures if f.done() and f.exception()), None)
            if failed:
                for f in futures:
                    f.cancel()
                job = jobs[futures.index(failed)]
                raise TimelapseError(
                    f"chunk {job.chunk.index} on {job.device} failed: {failed.exception()}"
                ) from failed.exception()
        drain(updates, timeout=0)
        if on_progress:
            on_progress(done)
        return [f.result() for f in futures]
