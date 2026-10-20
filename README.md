# timelapse

Turn long videos into time-lapses on Intel Arc GPUs. A 24-hour 4K recording becomes a 300x time-lapse in about 11 seconds on two Arc Pro B70s, or about 5 seconds when scaled to 1080p.

For each output frame, it seeks to the right point in the source and decodes only the frames it needs on the GPU. Each frame goes straight to the GPU encoder without leaving GPU memory, and it can be resized on the GPU on the way. The timeline is split into chunks that run in parallel across all Intel GPUs, and the chunks are then joined without re-encoding.

## Usage

Everything runs in a container, which provides an FFmpeg with VA-API and Intel's media driver. That FFmpeg is built for this job alone: it reads H.264, HEVC and AV1 from MP4, MOV and MKV files, and writes the same three containers.

```bash
docker compose build timelapse
docker compose run --rm timelapse video.mp4                       # → video_300x.mp4
docker compose run --rm timelapse a.mp4 b.mp4 --speedup 600       # several inputs
docker compose run --rm timelapse clip.mp4 --speedup 60 --fps 30 -o clip_60x.mp4
docker compose run --rm timelapse video.mp4 --scale 1920x1080     # 4K in, 1080p out
docker compose run --rm timelapse clip.mp4 --dry-run              # show the plan only
```

By default the current folder is mounted at `/data`. Set `VIDEOS=/path/to/videos` to use another folder. If your `render` group isn't GID 991, also set `RENDER_GID=$(getent group render | cut -d: -f3)`.

| Option | Default | |
|---|---|---|
| `--speedup` | 300 | How many times faster than real time the output plays. A whole number, at least 2 |
| `--fps` | input's rate | Output frame rate. It doesn't change the speedup, only the smoothness |
| `--scale` | input's size | Output size as `WxH`, resized on the GPU. `-1` on one side keeps the aspect ratio, e.g. `1920x-1`. See [Scaling](#scaling) |
| `-o`, `--output` | `<input>_<speedup>x.mp4` | Output file: `.mp4`, `.mov` or `.mkv`. MOV can't hold AV1. The folder must be writable, since the chunks are written there too |
| `--codec` | `hevc` | `hevc` (tagged `hvc1` for Apple players), `h264` or `av1` |
| `--quality` | 22 / 20 / 90 | CQP quality for hevc / h264 / av1; lower is better |
| `--exact` / `--keyframes` | auto | How frames are picked; see below |
| `--start`, `--end` | whole file | Source range in seconds |
| `--device` | all Intel GPUs | Render node(s), e.g. `/dev/dri/renderD128` |
| `--jobs-per-gpu` | 2 | Parallel chunks per GPU. One process can't keep a B70's media engines busy |

## How frames are picked

One output frame covers `speedup / fps` seconds of source. For example, 300x at 20 fps takes one frame every 15 s.

- **Nearest keyframe**: seeks and decodes a single keyframe per output frame, so it's the fastest. It's used when keyframes are close together compared with the step, or when the step is a whole number of keyframe intervals. With 1 s keyframes, any multiple of 20x at 20 fps qualifies.
- **Exact**: decodes forward from the keyframe before each point to the nearest frame. That's up to one GOP of extra decoding per output frame, with perfectly even timing. It's used otherwise, or with `--exact`.

Times on a 24 h 4K H.264 recording (16.67 fps, keyframes every 1.2 s, 40 GB) at 300x on two B70s. That's 4,798 output frames, or 4.8 minutes:

| Mode | 4K | `--scale 1920x1080` | Frames decoded |
|---|---|---|---|
| nearest keyframe | 10.7 s (1,464 MiB) | 5.2–5.3 s (595 MiB) | 4,798 |
| exact | 42.2 s (1,640 MiB) | 36.6 s (490 MiB) | 52,256 |

Scaled runs are faster because encoding 1080p frames is much cheaper than encoding 4K ones. At 4K, nearest-keyframe runs are limited by the encoder.

## Scaling

`--scale WxH` resizes each output frame with FFmpeg's `scale_vaapi` filter. Frames go from the decoder through the scaler to the encoder without leaving GPU memory. Scaling takes about 0.1 ms per frame, and the smaller encode more than makes up for it. Only the output frames are scaled, not every frame decoded on the way to them. With `-1` on one side, that side is computed from the input's aspect ratio, so `--scale 1920x-1` gives 1920x1080 for 16:9 input and 1920x1440 for 4:3.

On the B70, `scale_vaapi`'s `default`, `fast` and `hq` modes produce identical output. Scaling 4K camera frames to 1080p measured 43.2 dB PSNR against a software Lanczos scale, better than software bilinear (38.4 dB).

Limitations:

- **Exact size, no cropping or padding.** If `WxH` has a different aspect ratio from the input, the picture is stretched. The plan prints a warning when that happens. Use `-1` on one side to avoid it.
- **Upscales too.** Inputs smaller than `WxH` are scaled up. Inputs already at that size pass through unscaled.
- **One size per run.** With several inputs, each is scaled to the same `WxH`. A `-1` side is computed separately for each input.
- **Even sizes only.** 4:2:0 video needs even dimensions, so odd values are rejected. A side computed from `-1` is rounded to the nearest even number, which can be a pixel off the exact aspect ratio.
- **Stored pixels.** Sizes and aspect ratios use the frame's stored width and height. Rotation metadata and non-square pixels are ignored.
- **No color conversion.** The output keeps the input's pixel format and color tags, with no HDR tone mapping. It has been tested with 8-bit 4:2:0 H.264 input only.
- **No choice of algorithm.** The driver's scaler is the only one, A CPU scaler would cost about 10 ms per frame: 6.5 ms to copy each 4K frame out of GPU memory, plus 3–4.4 ms to scale it.
- **Tied to PyAV 18.1 and FFmpeg 8.** PyAV can't pass a GPU frames context to a filter graph, so `scale.py` makes that one libavfilter call through ctypes, using struct offsets pinned for these versions. With any other version, `--scale` refuses to run instead of guessing. After upgrading either, check the offsets: `tests/test_scale.py` checks the ones it can on the host, and the GPU tests cover the rest.

## Development

```bash
uv sync                                   # host: PyAV's PyPI wheel, enough for unit tests
uv run pytest                             # GPU tests are skipped on the host
uv run ruff check && uv run ruff format
docker compose build test timelapse       # rebuild after every change; the images contain the code
docker compose run --rm test              # everything, including GPU end-to-end tests
```

The GPU tests render synthetic videos whose frames have their frame number painted into the pixels. That lets them check exactly which source frame was picked for every output frame.

### Why a container, and why `av==18.1.0`

- **PyAV's PyPI wheels can't use these GPUs.** They bundle an FFmpeg built without VA-API, and on Linux FFmpeg's QSV support needs VA-API too. So the image compiles PyAV from source (`--no-binary-package av`) against its own FFmpeg.
- **The image builds FFmpeg 8.1 itself**, on Alpine 3.24, with only the decoders, encoders, filters and containers timelapse uses. Distribution builds link dozens of libraries for codecs, fonts and displays. Ubuntu 26.04's FFmpeg made the image 744 MB, and this build makes it 129 MB (48 MB compressed). To read another codec or container, add it to the `configure` flags in the Dockerfile's `ffmpeg` stage.
- **PyAV 19 requires FFmpeg 9**, hence the pin. To move to av 19+, bump `FFMPEG_VERSION` and its checksum in the Dockerfile, and re-check `scale.py`'s pinned offsets (see [Scaling](#scaling)).

### Layout

| Module | |
|---|---|
| `cli.py` | Typer CLI and Rich progress |
| `pipeline.py` | Planning; runs chunks in a process pool |
| `worker.py` | Per-chunk seek, GPU decode, GPU scale, GPU encode, writing an MP4 segment |
| `scale.py` | `scale_vaapi` filter graphs for GPU frames, and the output-size math |
| `concat.py` | Joins segments into the final MP4 |
| `probe.py` | Video info and keyframe spacing (demuxing only) |
| `sampling.py` | Pure logic: sample grid, chunks, mode choice |
| `gpu.py` | Finds Intel render nodes; checks VA-API encoding |
