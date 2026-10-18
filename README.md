# timelapse

Turn long videos into time-lapses on Intel Arc GPUs. A 3.6-hour 4K recording becomes a 300x time-lapse in about 2 seconds on two Arc Pro B70s.

For each output frame, it seeks to the right point in the source and decodes only the frames it needs on the GPU. Each frame goes straight to the GPU encoder without leaving GPU memory. The timeline is split into chunks that run in parallel across all Intel GPUs, and the chunks are then joined without re-encoding.

## Usage

Everything runs in a container, which provides an FFmpeg with VA-API and Intel's media driver:

```bash
docker compose build timelapse
docker compose run --rm timelapse fixed_input.mp4                 # → fixed_input_300x.mp4
docker compose run --rm timelapse a.mp4 b.mp4 --speedup 600       # several inputs
docker compose run --rm timelapse clip.mp4 --speedup 60 --fps 30 -o clip_60x.mp4
docker compose run --rm timelapse clip.mp4 --dry-run              # show the plan only
```

By default the current folder is mounted at `/data`. Set `VIDEOS=/path/to/videos` to use another folder. If your `render` group isn't GID 991, also set `RENDER_GID=$(getent group render | cut -d: -f3)`.

| Option | Default | |
|---|---|---|
| `--speedup` | 300 | How many times faster than real time the output plays |
| `--fps` | input's rate | Output frame rate. It doesn't change the speedup, only the smoothness |
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

Times on a 3.6 h 4K H.264 file with 1 s keyframes, at 300x on two B70s:

| Mode | Time | Frames decoded |
|---|---|---|
| nearest keyframe | 2.2 s | 869 |
| exact | 8.9 s | 10,417 |

## Development

```bash
uv sync                                   # host: PyAV's PyPI wheel, enough for unit tests
uv run pytest                             # GPU tests are skipped on the host
uv run ruff check && uv run ruff format
docker compose run --rm test              # everything, including GPU end-to-end tests
```

The GPU tests render synthetic videos whose frames have their frame number painted into the pixels. That lets them check exactly which source frame was picked for every output frame.

### Why a container, and why `av==18.1.0`

- **PyAV's PyPI wheels can't use these GPUs.** They bundle an FFmpeg built without VA-API, and on Linux FFmpeg's QSV support needs VA-API too. So the image compiles PyAV from source against Ubuntu 26.04's FFmpeg (`--no-binary-package av`).
- **PyAV 19 requires FFmpeg 9**, while Ubuntu 26.04 ships FFmpeg 8.0, hence the pin. Move to av 19+ when the base image has FFmpeg 9.

### Layout

| Module | |
|---|---|
| `cli.py` | Typer CLI and Rich progress |
| `pipeline.py` | Planning; runs chunks in a process pool |
| `worker.py` | Per-chunk seek, GPU decode, GPU encode, writing an MP4 segment |
| `concat.py` | Joins segments into the final MP4 |
| `probe.py` | Video info and keyframe spacing (demuxing only) |
| `sampling.py` | Pure logic: sample grid, chunks, mode choice |
| `gpu.py` | Finds Intel render nodes; checks VA-API encoding |
