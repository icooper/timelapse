# syntax=docker/dockerfile:1

# FFmpeg 8 with VA-API, and Intel's media driver for Arc GPUs.
FROM ubuntu:26.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg intel-media-va-driver-non-free vainfo \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev pkg-config \
        libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
        libswresample-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /src
# PyAV is compiled against the image's FFmpeg: the FFmpeg bundled in its PyPI
# wheels has no VA-API, so it can't use the GPUs.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --no-binary-package av
COPY README.md ./
COPY src ./src
# --reinstall-package: uv decides whether to rebuild a local project from
# pyproject.toml's mtime, so without it the cache mount can serve a stale wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --no-binary-package av --reinstall-package timelapse

FROM build AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-binary-package av
COPY tests ./tests
ENTRYPOINT ["/opt/venv/bin/pytest"]

FROM base AS runtime
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /data
ENTRYPOINT ["timelapse"]
