"""Find Intel GPUs and check that they can encode with VA-API."""

from __future__ import annotations

import io
from pathlib import Path

import av
from av.codec.hwaccel import HWAccel, hwdevices_available

INTEL_VENDOR = "0x8086"
ENCODERS = {"hevc": "hevc_vaapi", "h264": "h264_vaapi", "av1": "av1_vaapi"}


def discover_intel_gpus(drm: Path = Path("/sys/class/drm"), dev: Path = Path("/dev/dri")) -> list[str]:
    """Render nodes of Intel GPUs that are present (and, in a container, passed through)."""
    nodes = []
    for node in sorted(drm.glob("renderD*"), key=lambda p: int(p.name.removeprefix("renderD"))):
        try:
            vendor = (node / "device" / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor == INTEL_VENDOR and (dev / node.name).exists():
            nodes.append(str(dev / node.name))
    return nodes


def check_device(device: str, encoder: str) -> str | None:
    """Encode one small frame on the device. Return an error message, or None if it works."""
    if "vaapi" not in hwdevices_available():
        return "this PyAV build has no VA-API support; run timelapse inside its container"
    try:
        with av.open(io.BytesIO(), "w", format="nut") as out:
            stream = out.add_stream(encoder, rate=1, hwaccel=HWAccel("vaapi", device))
            stream.width, stream.height, stream.pix_fmt = 640, 360, "nv12"
            frame = av.VideoFrame(640, 360, "nv12")
            frame.pts = 0
            out.mux(stream.encode(frame))
            out.mux(stream.encode(None))
    except (av.FFmpegError, NotImplementedError, RuntimeError) as e:
        return str(e)
    return None
