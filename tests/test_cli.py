from fractions import Fraction
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner
from videos import make_video

from timelapse.cli import app, default_output, parse_rate
from timelapse.probe import keyframe_stats, probe


@pytest.mark.parametrize(
    "text,rate", [("25", 25), ("29.97", Fraction(2997, 100)), ("30000/1001", Fraction(30000, 1001))]
)
def test_parse_rate(text, rate):
    assert parse_rate(text) == rate


@pytest.mark.parametrize("text", ["0", "-5", "abc", "1/0"])
def test_parse_rate_rejects_bad_values(text):
    with pytest.raises(typer.BadParameter):
        parse_rate(text)


def test_default_output_names_by_speedup():
    assert default_output(Path("/v/clip.mp4"), 300) == Path("/v/clip_300x.mp4")
    assert default_output(Path("clip.mov"), 62.5) == Path("clip_62.5x.mp4")


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


def test_output_with_several_inputs_is_rejected(clip):
    result = CliRunner().invoke(app, [str(clip), str(clip), "-o", "out.mp4", "--dry-run"])
    assert result.exit_code != 0
