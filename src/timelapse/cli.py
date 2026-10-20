"""Command-line interface."""

from __future__ import annotations

import os
import time
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from timelapse.concat import OUTPUT_FORMATS, UNSUPPORTED_CODECS
from timelapse.errors import TimelapseError
from timelapse.gpu import ENCODERS, check_device, discover_intel_gpus
from timelapse.pipeline import Plan, Settings, make_plan, run_plan
from timelapse.scale import Size, unsupported_reason

console = Console(stderr=True, highlight=False)
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


class Codec(StrEnum):
    hevc = "hevc"
    h264 = "h264"
    av1 = "av1"


def parse_rate(text: str) -> Fraction:
    try:
        rate = Fraction(text)
    except ValueError, ZeroDivisionError:
        raise typer.BadParameter(f"invalid frame rate {text!r}") from None
    if rate <= 0:
        raise typer.BadParameter("frame rate must be positive")
    return rate


def parse_size(text: str) -> Size:
    try:
        width, height = (int(side) for side in text.lower().split("x"))
    except ValueError:
        raise typer.BadParameter(f"invalid size {text!r}; expected WxH, such as 1920x1080") from None
    if (width, height) == (-1, -1) or any(side != -1 and (side <= 0 or side % 2) for side in (width, height)):
        raise typer.BadParameter("width and height must be positive even numbers, or -1 for one of them")
    return Size(width, height)


def default_output(source: Path, speedup: int) -> Path:
    return source.with_name(f"{source.stem}_{speedup}x.mp4")


def check_output_name(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        raise typer.BadParameter(f"{path} is a folder; name the output file, e.g. {path / 'timelapse.mp4'}")
    if path.suffix.lower() not in OUTPUT_FORMATS:
        kind = f"{path.suffix} files" if path.suffix else "files without an extension"
        raise typer.BadParameter(f"can't write {kind}; use {either(OUTPUT_FORMATS)}")
    return path


def check_output_codec(output: Path, codec: str) -> None:
    if codec in UNSUPPORTED_CODECS.get(output.suffix.lower(), ()):
        fits = [suffix for suffix in OUTPUT_FORMATS if codec not in UNSUPPORTED_CODECS.get(suffix, ())]
        raise typer.BadParameter(f"{output.suffix} files can't hold {codec}; use {either(fits)}")


def either(choices) -> str:
    *others, last = choices
    return f"{', '.join(others)} or {last}" if others else last


def check_writable(output: Path) -> None:
    """Fail before rendering, not after: segments and the output go in the output's folder."""
    folder = output.resolve().parent
    if not folder.exists():
        raise TimelapseError(f"can't write to {folder}: it doesn't exist")
    if not folder.is_dir():
        raise TimelapseError(f"can't write to {folder}: it isn't a folder")
    if not os.access(folder, os.W_OK):
        if os.statvfs(folder).f_flag & os.ST_RDONLY:
            raise TimelapseError(f"can't write to {folder}: it's read-only (in Docker, mount it without :ro)")
        raise TimelapseError(f"can't write to {folder}: this user doesn't have permission")


class RateColumn(ProgressColumn):
    def render(self, task) -> Text:
        elapsed = task.elapsed or 0
        rate = f"{task.completed / elapsed:6.1f} fps" if elapsed > 0 else "     - fps"
        return Text(rate, style="progress.data.speed")


def describe(plan: Plan) -> None:
    info, kf, step = plan.info, plan.keyframes, float(plan.step)
    console.print(
        f"🎥 {plan.source.name}: {info.width}x{info.height} {info.codec}, "
        f"{float(info.frame_rate):.4g} fps nominal ({info.avg_fps:.2f} avg), {info.duration:.1f}s"
    )
    if info.avg_fps and step * info.avg_fps < 1:
        console.print("⚠️  sampling faster than the source frame rate, so frames will repeat")
    if plan.size != Size(info.width, info.height):
        console.print(f"📐 scaling {info.width}x{info.height} → {plan.size} on the GPU")
        if abs(plan.size.width * info.height / (plan.size.height * info.width) - 1) > 0.01:
            console.print("⚠️  that changes the aspect ratio; use -1 for one side to keep it, e.g. 1920x-1")
    if kf.worst == float("inf"):
        kf_text = "keyframes too sparse to measure"
    else:
        kf_text = f"keyframes every {kf.typical:.2f}s (max {kf.worst:.2f}s)"
    if plan.keyframes_only:
        mode = f"nearest keyframe ({step:.4g}s step ≈ {step / kf.typical:.2f} keyframe intervals)"
    else:
        mode = "exact (nearest frame, decoding forward from the keyframe before it)"
    console.print(f"🔑 {kf_text} → {mode}")
    devices = ", ".join(Path(d).name for d in plan.devices)
    console.print(
        f"🧩 1 frame per {step:.4g}s of source → {plan.frames} frames, "
        f"{float(plan.frames / plan.fps):.1f}s at {float(plan.fps):.4g} fps, "
        f"{len(plan.chunks)} chunks on {devices}"
    )


def show_chunks(plan: Plan) -> None:
    table = Table("chunk", "device", "from", "to", "frames", box=None)
    for chunk in plan.chunks:
        table.add_row(
            str(chunk.index),
            Path(plan.device_for(chunk)).name,
            f"{chunk.targets[0]:.1f}s",
            f"{chunk.targets[-1]:.1f}s",
            str(chunk.frames),
        )
    console.print(table)


def render(plan: Plan, keep_temp: bool) -> None:
    started = time.perf_counter()
    columns = [
        TextColumn("{task.description:>12}"),
        BarColumn(),
        MofNCompleteColumn(),
        RateColumn(),
        TimeElapsedColumn(),
    ]
    with Progress(*columns, console=console) as progress:
        tasks = {}
        for device in plan.devices:
            total = sum(c.frames for c in plan.chunks if plan.device_for(c) == device)
            tasks[device] = progress.add_task(Path(device).name, total=total)
        overall = progress.add_task("total", total=plan.frames)

        def update(done: dict[int, int]) -> None:
            for device, task in tasks.items():
                completed = sum(n for i, n in done.items() if plan.device_for(plan.chunks[i]) == device)
                progress.update(task, completed=completed)
            progress.update(overall, completed=sum(done.values()))

        results = run_plan(plan, keep_temp=keep_temp, on_progress=update)
    elapsed = time.perf_counter() - started
    size = plan.output.stat().st_size / 2**20
    decoded = sum(r.decoded for r in results)
    console.print(
        f"🎉 Saved {plan.output} ({size:.0f} MiB) in {elapsed:.1f}s, decoding {decoded} source frames"
    )


@app.command()
def timelapse(
    inputs: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, help="Video files to speed up.")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            callback=check_output_name,
            help="Output file: .mp4, .mov or .mkv (default: <input>_<speedup>x.mp4).",
        ),
    ] = None,
    speedup: Annotated[int, typer.Option(min=2, help="How many times faster than real time.")] = 300,
    fps: Annotated[
        Fraction | None,
        typer.Option(parser=parse_rate, metavar="RATE", help="Output frame rate (default: the input's)."),
    ] = None,
    scale: Annotated[
        Size | None,
        typer.Option(
            parser=parse_size,
            metavar="WxH",
            help="Resize on the GPU, e.g. 1920x1080. -1 on one side keeps the aspect ratio (1920x-1).",
        ),
    ] = None,
    codec: Annotated[Codec, typer.Option(help="Output codec.")] = Codec.hevc,
    quality: Annotated[
        int | None, typer.Option(help="CQP quality, lower is better (default: hevc 22, h264 20, av1 90).")
    ] = None,
    device: Annotated[
        list[str] | None, typer.Option(help="Render node to use, repeatable (default: all Intel GPUs).")
    ] = None,
    jobs_per_gpu: Annotated[int, typer.Option(min=1, help="Parallel chunks per GPU.")] = 2,
    exact: Annotated[bool, typer.Option("--exact", help="Always pick the exact nearest frame.")] = False,
    keyframes: Annotated[
        bool, typer.Option("--keyframes", help="Always pick the nearest keyframe, even if timing is uneven.")
    ] = False,
    start: Annotated[float, typer.Option(min=0, help="Source start time in seconds.")] = 0.0,
    end: Annotated[float | None, typer.Option(help="Source end time in seconds.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the plan without rendering.")] = False,
    keep_temp: Annotated[bool, typer.Option("--keep-temp", help="Keep the per-chunk segment files.")] = False,
) -> None:
    """Turn long videos into time-lapses on Intel Arc GPUs."""
    if output is not None and len(inputs) > 1:
        raise typer.BadParameter("--output only works with a single input")
    if exact and keyframes:
        raise typer.BadParameter("--exact and --keyframes can't be combined")
    settings = Settings(
        speedup=speedup,
        fps=fps,
        scale=scale,
        codec=codec.value,
        quality=quality,
        jobs_per_gpu=jobs_per_gpu,
        mode="exact" if exact else "keyframes" if keyframes else "auto",
        start=start,
        end=end,
    )
    outputs = [output or default_output(source, speedup) for source in inputs]
    for target in outputs:
        check_output_codec(target, settings.codec)
    try:
        for target in outputs:
            check_writable(target)
        devices = device or discover_intel_gpus()
        if not devices:
            raise TimelapseError("no Intel GPU render nodes found; pass --device or check /dev/dri")
        if not dry_run:
            for d in devices:
                if error := check_device(d, ENCODERS[settings.codec]):
                    raise TimelapseError(f"{d} can't encode with {ENCODERS[settings.codec]}: {error}")
            if scale and (reason := unsupported_reason()):
                raise TimelapseError(f"--scale: {reason}")
        for source, target in zip(inputs, outputs, strict=True):
            plan = make_plan(source, target, settings, devices)
            describe(plan)
            if dry_run:
                show_chunks(plan)
            else:
                render(plan, keep_temp)
    except TimelapseError as e:
        console.print(f"[bold red]❌ {e}[/]")
        raise typer.Exit(1) from None


def main() -> None:
    app()
