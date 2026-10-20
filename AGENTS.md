# AGENTS.md

Notes for coding agents. README.md is the user's documentation. This file covers what you need to change the code safely: commands, environment, invariants, and the gotchas that have already cost time.

## What this is

`timelapse` is a Python 3.14 CLI (Typer and Rich) that turns long 4K camera recordings into time-lapses on two Intel Arc Pro B70 GPUs, using PyAV and VA-API. Frames are decoded, optionally scaled, and encoded on the GPU, and they never reach system memory.

The pipeline:
1. `probe` reads the video's properties.
2. `make_plan` lays out the sample grid and chunks.
3. A process pool runs `len(gpus) × jobs_per_gpu` workers.
4. Each worker runs `FramePicker` → `GpuScaler` → encoder → an MP4 segment.
5. `concat_segments` remuxes the segments into the output, without re-encoding.

The module table is in README.md under Layout.

## Commands

```bash
uv sync
uv run pytest                                    # host: unit tests only, GPU tests skip
uv run ruff check
uv run ruff format --check
docker compose build test timelapse              # the images contain the code: rebuild after every change
RENDER_GID=$(getent group render | cut -d: -f3) docker compose run --rm test -q   # all tests, on the GPU
```

Run lint, the build and the container tests as separate commands, not one `&&` chain. Otherwise a lint failure skips the build, and you end up testing an old image.

To run a real render, mount the videos read-only and write the output somewhere else:

```bash
docker run --rm --device /dev/dri --group-add 991 -u 1000:1000 \
  -v "$PWD":/data:ro -v /tmp/out:/out timelapse:latest basement.mp4 -o /out/test.mp4 [--scale 1920x1080] [--exact]
```

## Environment

- **GPUs:** `/dev/dri/renderD128` and `renderD130` are the Arcs. `renderD129` is the AMD iGPU, so never use it. `discover_intel_gpus()` filters by vendor `0x8086`. The render group's GID is 991.
- **Anything that touches the GPU must run in the container.** The host's `.venv` uses PyAV's PyPI wheel, whose bundled FFmpeg has no VA-API. QSV on Linux needs VA-API too.
- **The image is Alpine 3.24 (musl) with its own FFmpeg 8.1.3**, built in the Dockerfile's `ffmpeg` stage with only the components timelapse uses. See Build below.
- **PyAV is pinned to `av==18.1.0`.** The image builds it from source (`--no-binary-package av`) against that FFmpeg. av 19 needs FFmpeg 9.
- **Videos** (`*.mp4`, gitignored) sit in the repo root and belong to the user. Never modify, move or delete them. They get renamed and deleted often, so run `ls` before relying on one.

## Benchmarks

- **Use `basement.mp4`:** a 24 h, 3840x2160 H.264 recording, 16.67 fps nominal (17.8 average), keyframes every 1.2 s, 40 GB.
  - At the default 300x it gives 4,798 output frames.
  - A 4K output is about 1.5 GB. Write outputs to a temp dir and delete them afterwards.
- **Do a warm-up run first**, then run each configuration 2–3 times.
- **Current numbers** are in README.md's table under "How frames are picked":
  - Nearest keyframe: 10.7 s at 4K (encode-bound), 5.2–5.3 s with `--scale 1920x1080`.
  - Exact: 42.2 s at 4K, 36.6 s at 1080p.
- **Update that table** whenever performance changes.
- **Per-stage timing:** wrap `FramePicker.pick` and `GpuScaler.__call__` in a script that calls `render_chunk` directly.
- **Comparing implementations:** the CLI's 🎉 line times the render. Hash the outputs too. The decoder changes so far left them byte-identical.
  - Across FFmpeg builds, hash the video packets, not the file. The MP4's `encoder` tag carries libavformat's version. The move from Ubuntu's FFmpeg 8.0 to our 8.1.3 left the packets identical.

## Invariants and gotchas

**The GPU frame path**
- **Zero copy:**
  - The decoder uses `HWAccel(..., allow_software_fallback=False, is_hw_owned=True)`.
  - The encoder gets `pix_fmt = "vaapi"` and its own `HWAccel`, then adopts the first frame's frames context.
  - Keep `reformat()` and `to_ndarray()` out of the hot path. Copying one 4K frame out of the GPU costs about 6.5 ms.
- **Set `frame.pict_type = PictureType.NONE` before encoding.** Decoded keyframes keep type I, which forces IDR frames. Without the reset, every output frame was an I-frame and the file was 1.7x larger.
- **On FFmpeg 8.0, a flushed H.264 decoder made a new frames context (pool of GPU frames).** FFmpeg rebuilt the hwaccel on the next frame, at about 4.5 ms per 4K seek. FFmpeg 8.1 keeps the pool across flushes, with or without B-frames.
  - `container.seek()` flushes every stream's own decoder. So the worker decodes with its own `Decoder`, which seeks don't touch.
  - `Decoder` flushes only after draining, or before going back to a packet it has already decoded. It drops frames held from before a seek.
- **Set the decoder's `pix_fmt` to a software format before the first packet.** With `hwaccel=`, PyAV presets it to `vaapi`. FFmpeg's `av1` decoder takes that as an hwaccel already set up, skips `get_format()`, and fails with "get_buffer() failed" (`avcodec_send_packet()` returns 22). `Decoder` and the tests' `decode_indices()` both reset it. H.264 and HEVC aren't affected.
- **Streams with B-frames** hold an isolated keyframe back for reordering, and `low_delay` doesn't help.
  - Keyframe mode drains them with `decode(None)`, so the next pick flushes. Drained frames have no `time_base`.
  - The cameras have no B-frames. The synthetic test videos have them, except `gop35_ipp`.
- **Frames contexts:**
  - Never key a cache on a frames-context pointer. `GpuScaler` used to, and rebuilt its graph for every frame, at about 4 ms each.
  - When counting pools, keep the frames alive, or freed addresses get reused and you undercount. Compare the `AVHWFramesContext` address, as `frames_context()` in the GPU tests does. The frame's `hw_frames_ctx` reference differs for every frame.
- **`hevc_vaapi` uses B-frames**, so pts ≠ dts. The worker checks that timestamps stay valid.
  - Segments must be MP4: NUT loses dts on demux.
  - `concat` needs `add_stream_from_template(stream, opaque=True)`. Without `opaque`, PyAV looks up an encoder by the decoder's name, such as `av1`, and fails.

**`scale.py` (the ctypes bridge)**
- PyAV 18 can't give a filter graph's buffer source a hardware frames context. So `scale.py` calls `av_buffersrc_parameters_set` through ctypes, on the libavfilter PyAV already loaded.
- **It reads raw pointers at pinned offsets:**
  - the Cython object layouts of PyAV 18.1;
  - the FFmpeg 8 struct offsets for libavutil 60 and libavfilter 11.
- **`unsupported_reason()` refuses other versions.**
- **On an upgrade, recompute the offsets.** Compile a small C program in the `build` stage, against the headers in `/usr/local/include`, that prints `offsetof` for:
  - `AVFrame` width, height, format and hw_frames_ctx;
  - `AVHWFramesContext` format and sw_format;
  - `AVBufferSrcParameters` format and hw_frames_ctx;
  - `AVFilterContext` name.

  Then update the constants, and run both the host and GPU tests.
- **A PyAV `FilterContext` doesn't keep its `Graph` alive.** Keep a reference to the graph, or reading through the context segfaults.
- **Graphs are rebuilt only when the frames' sw_format or size changes.** Surfaces from any pool on the same device work in an existing graph.

**Sampling and modes**
- `step = speedup / fps` is an exact `Fraction`. Targets are `start + k*step`, so the grid never drifts.
- `--speedup` is an integer ≥ 2.
- Chunks are contiguous, and the tests check that they join with no duplicate or missing frames.
- **Auto mode** (`keyframes_suffice`) uses keyframes when the worst gap is ≤ step/4, or when the step is within 0.1 of a whole number of keyframe intervals.
- **User decisions, don't change:**
  - `--exact` stays opt-in.
  - `--scale` stays an exact `WxH`: it upscales, and it stretches with a warning.
  - `--jobs-per-gpu` stays 2 by default. A sweep on `basement.mp4` after the `Decoder` change gave these times:

    | Jobs per GPU | Keyframes, 4K | Keyframes, 1080p | Exact, 4K | Exact, 1080p |
    |---|---|---|---|---|
    | 1 | 16.2 s | 9.1 s | 64.0 s | 64.2 s |
    | 2 | 10.8 s | 5.3 s | 42.1 s | 36.6 s |
    | 3 | 10.7 s | 4.8 s | 44.3 s | 38.6 s |
    | 4 | 10.7 s | 4.8 s | 48.8 s | 42.8 s |
    | 6 | 10.7 s | 4.8 s | 44.4 s | 38.5 s |

**Build**
- **FFmpeg is built with `--disable-everything`,** then each component is enabled by name. Anything missing fails at runtime, often with an unhelpful error. The list of what's enabled is in the build log's configure summary (`docker build --progress=plain`).
  - Enabled for tests or helpers, not renders: `libx264` for `make_video()`, and the NUT muxer for `check_device()`.
  - The output muxers must match `OUTPUT_FORMATS` in `concat.py`, which the CLI checks `-o` against. `UNSUPPORTED_CODECS` lists the codecs a container can't hold (AV1 in MOV).
  - `buffer` and `buffersink` are always built, so the summary lists only `scale_vaapi` under filters.
  - It builds in about 20 s. PyAV rebuilds whenever the `ffmpeg` stage changes, which takes about 80 s.
- **Bumping `FFMPEG_VERSION`:** check the tarball's `.asc` signature against FFmpeg's release key (`FCF986EA15E6E293A5644F10B4322F04D67658D8`), update the `ADD --checksum`, and recompute the `scale.py` offsets.
- **Keep `--refresh-package av` and the `uv-alpine` cache id.** uv tags the PyAV wheel it builds `linux_x86_64`, whatever libc or FFmpeg it was built against, and reuses it from the cache. A glibc wheel left over from the Ubuntu image once got installed on Alpine. PyAV then fell back to its uncompiled `.py` sources and failed with `No module named 'cython'`.
- **Keep the `strip` after the PyAV build.** Alpine's Python compiles extensions with `-g`, and stripping saves 25 MB.
- **Keep `--reinstall-package timelapse` in the Dockerfile.** uv decides whether to rebuild the project from `pyproject.toml`'s mtime, so without it the cache mount can serve a stale wheel. This once made the tests run old code.
- **`.dockerignore` is an allowlist** that keeps the 40 GB of videos out of the build context. Add any new top-level file the image needs.

## Tests

- **`tests/videos.py`** paints each frame's index into its luma. `decode_indices()` reads the indices back, so tests check exactly which source frame was picked.
  - AV1 outputs need `decode_indices(path, gpu_device)`: the image has no software AV1 decoder.
  - `make_video(..., bframes=False)` gives a stream like the cameras', which takes `Decoder`'s no-flush path. The B-frame videos cover draining and the flushes.
- **GPU tests** are marked `gpu`. The `gpu_device` fixture skips them unless VA-API encoding works.
  - Any change to `worker.py`, `scale.py` or `concat.py` needs a GPU test.
  - Pure logic gets a host test.
- **Warnings raised in worker processes don't reach pytest.** To catch deprecations, run the code in the test process, as `test_scaler_keeps_its_graph_across_pools` does, together with `-W error::DeprecationWarning`.

## Conventions

- **Ruff:** line length 110, rules `E F I UP B`, target py314. `except ValueError, ZeroDivisionError:` without parentheses is valid 3.14 syntax, so leave it.
- **Errors:** the user sees `TimelapseError`, which the CLI prints as `❌ …` before exiting with code 1. Invalid arguments use `typer.BadParameter`.
  - The maintainer wants a clean, informative message for every failure the user can cause, never a traceback. Say what's wrong and what to do instead.
  - Check what you can before rendering starts. A bad output used to fail only in `concat`, after the whole render. The CLI now checks the output's extension, its codec and its folder up front.
- **CLI output** is one line per fact, each starting with an emoji: 🎥 🔑 📐 🧩 ⚠️ 🎉 ❌.
- **Comments** explain why, usually an FFmpeg or PyAV quirk, not what. Match the existing density.

## Working agreements

- Don't commit or push. The maintainer commits.
- Keep README.md current with every user-visible change: the options table, limitations and benchmark numbers.
- Work autonomously and make reasonable choices. Collect open questions for the end.

## Known opportunities

- **Moving to PyAV 19+ and FFmpeg 9:** bump `FFMPEG_VERSION` (see Build), then redo `scale.py`'s offsets. Check upstream too: PyAV main's `Graph` now accepts a `hw_device`, and it may grow buffer-source frames-context support, which would replace the ctypes bridge.
- **`Decoder` may no longer be needed.** It avoids the pool rebuild that flushes caused on FFmpeg 8.0, and 8.1 no longer rebuilds the pool. Benchmark decoding through the container's own decoder before removing it: flushes may still cost something.
