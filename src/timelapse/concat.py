"""Join per-chunk segments into the final MP4 without re-encoding."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av

# Output containers by file extension. The image's FFmpeg has only these muxers.
OUTPUT_FORMATS = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska"}
# FFmpeg's MOV muxer has no tag for AV1.
UNSUPPORTED_CODECS = {".mov": {"av1"}}


def concat_segments(segments: list[Path], output: Path, fps: Fraction, codec: str) -> int:
    """Remux the segments back to back into `output`. Returns the number of frames written."""
    frame_time = 1 / Fraction(fps)
    total = 0
    muxer = OUTPUT_FORMATS[output.suffix.lower()]
    with av.open(str(output), "w", format=muxer, container_options={"movflags": "+faststart"}) as out:
        out_stream = None
        for path in segments:
            with av.open(str(path)) as segment:
                stream = segment.streams.video[0]
                if out_stream is None:
                    out_stream = out.add_stream_from_template(stream, opaque=True)  # remux: no encoder lookup
                    out_stream.time_base = frame_time
                    if codec == "hevc":
                        out_stream.codec_tag = "hvc1"  # lets QuickTime / Apple devices play it
                # A segment starts with an IDR frame carrying its lowest pts. Rebase on
                # it so the segment follows the frames already written; with B-frames
                # its first dts lands just before that, after the previous segment's last.
                start = round(total * frame_time / stream.time_base)
                base = None
                for packet in segment.demux(stream):
                    if packet.pts is None:  # end of stream
                        continue
                    if base is None:
                        base = packet.pts
                    packet.pts += start - base
                    packet.dts += start - base
                    packet.stream = out_stream
                    out.mux(packet)
                    total += 1
    return total
