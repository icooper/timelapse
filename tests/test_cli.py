from fractions import Fraction
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner
from videos import make_video

from timelapse.cli import app, default_output, parse_rate, parse_size
from timelapse.probe import keyframe_stats, probe
from timelapse.scale import Size


@pytest.mark.parametrize(
    "text,rate", [("25", 25), ("29.97", Fraction(2997, 100)), ("30000/1001", Fraction(30000, 1001))]
)
def test_parse_rate(text, rate):
    assert parse_rate(text) == rate


@pytest.mark.parametrize("text", ["0", "-5", "abc", "1/0"])
def test_parse_rate_rejects_bad_values(text):
    with pytest.raises(typer.BadParameter):
        parse_rate(text)


@pytest.mark.parametrize(
    "text,size", [("1920x1080", Size(1920, 1080)), ("1920X-1", Size(1920, -1)), ("-1x720", Size(-1, 720))]
)
def test_parse_size(text, size):
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["1920", "1920x1080x2", "wxh", "0x1080", "1921x1080", "-1x-1", "-2x720"])
def test_parse_size_rejects_bad_values(text):
    with pytest.raises(typer.BadParameter):
        parse_size(text)


def test_default_output_names_by_speedup():
    assert default_output(Path("/v/clip.mp4"), 300) == Path("/v/clip_300x.mp4")
    assert default_output(Path("clip.mov"), 60) == Path("clip_60x.mp4")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("video") / "clip.mp4", seconds=60, fps=20, gop=20)


def test_probe_and_keyframe_stats(clip):
    info = probe(clip)
    assert (info.width, info.height, info.codec, info.frame_rate) == (640, 360, "h264", 20)
    assert info.duration == pytest.approx(60, abs=0.1)
    kf = keyframe_stats(clip, info)
    assert kf.typical == pytest.approx(1.0)
    assert kf.worst == pytest.approx(1.0)


def test_dry_run_shows_plan_without_touching_the_gpu(clip):
    result = CliRunner().invoke(
        app,
        [
            str(clip),
            "--speedup",
            "100",
            "--device",
            "/dev/dri/renderD128",
            "--device",
            "/dev/dri/renderD130",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "12 frames" in result.output  # 60s / (100 / 20 fps) = 12 samples
    assert "nearest keyframe" in result.output
    assert "4 chunks" in result.output


def dry_run(clip, *args: str):
    return CliRunner().invoke(app, [str(clip), *args, "--device", "/dev/dri/renderD128", "--dry-run"])


@pytest.mark.parametrize("speedup", ["1", "0", "-5", "2.5", "abc"])
def test_speedup_must_be_an_integer_above_one(clip, speedup):
    result = dry_run(clip, "--speedup", speedup)
    assert result.exit_code == 2, result.output
    assert "--speedup" in result.output


@pytest.mark.parametrize("size", ["320x-1", "-1x180"])
def test_dry_run_shows_scaling(clip, size):
    result = dry_run(clip, "--scale", size)
    assert result.exit_code == 0, result.output
    assert "scaling 640x360 → 320x180" in result.output
    assert "aspect ratio" not in result.output


def test_dry_run_warns_when_scaling_stretches(clip):
    result = dry_run(clip, "--scale", "320x320")
    assert result.exit_code == 0, result.output
    assert "changes the aspect ratio" in result.output


def test_output_with_several_inputs_is_rejected(clip):
    result = CliRunner().invoke(app, [str(clip), str(clip), "-o", "out.mp4", "--dry-run"])
    assert result.exit_code != 0
