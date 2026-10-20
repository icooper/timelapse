# syntax=docker/dockerfile:1

# Python, Intel's media driver for Arc GPUs, and the libraries our FFmpeg links.
FROM alpine:3.24 AS base
RUN apk add --no-cache python3 intel-media-driver libva libdrm x264-libs

# FFmpeg with only what timelapse uses: VA-API decoding, scaling and encoding,
# and MP4 in and out. Distribution builds link dozens of codec, font and
# display libraries. libx264 is only there for the tests' synthetic videos, and
# the NUT muxer for check_device().
FROM base AS ffmpeg
RUN apk add --no-cache build-base nasm pkgconf libva-dev libdrm-dev x264-dev
ARG FFMPEG_VERSION=8.1.3
ADD --checksum=sha256:7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3 \
    https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz /tmp/
RUN tar -xf /tmp/ffmpeg-${FFMPEG_VERSION}.tar.xz -C /tmp \
    && cd /tmp/ffmpeg-${FFMPEG_VERSION} \
    && ./configure --prefix=/usr/local \
        --enable-shared --disable-static --disable-programs --disable-doc --disable-debug \
        --disable-autodetect --disable-network --disable-everything \
        --enable-gpl --enable-libx264 --enable-vaapi --enable-libdrm \
        --enable-protocol=file \
        --enable-demuxer=mov,matroska \
        --enable-muxer=mp4,mov,matroska,nut \
        --enable-parser=h264,hevc,av1 \
        --enable-decoder=h264,hevc,av1 \
        --enable-hwaccel=h264_vaapi,hevc_vaapi,av1_vaapi \
        --enable-encoder=h264_vaapi,hevc_vaapi,av1_vaapi,libx264 \
        --enable-filter=buffer,buffersink,scale_vaapi \
    && make -j"$(nproc)" install \
    && rm -rf /tmp/ffmpeg-*

FROM ffmpeg AS build
RUN apk add --no-cache python3-dev
COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /bin/uv
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
WORKDIR /src
# PyAV is compiled against our FFmpeg: the FFmpeg bundled in its PyPI wheels
# has no VA-API, so it can't use the GPUs. The wheel uv builds is tagged
# linux_x86_64 whatever libc or FFmpeg it was built against, so uv would reuse
# one built for another FFmpeg: --refresh-package rebuilds it. The cache id
# keeps it apart from caches the glibc images used. Alpine's Python compiles
# extensions with -g, and stripping takes PyAV from 29 MB to 5 MB.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,id=uv-alpine,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --no-binary-package av --refresh-package av \
    && find /opt/venv -name '*.so' -exec strip --strip-unneeded {} +
COPY README.md ./
COPY src ./src
# --reinstall-package: uv decides whether to rebuild a local project from
# pyproject.toml's mtime, so without it the cache mount can serve a stale wheel.
RUN --mount=type=cache,id=uv-alpine,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --no-binary-package av --reinstall-package timelapse

FROM build AS test
RUN --mount=type=cache,id=uv-alpine,target=/root/.cache/uv \
    uv sync --locked --no-binary-package av
COPY tests ./tests
ENTRYPOINT ["/opt/venv/bin/pytest"]

FROM base AS runtime
COPY --from=ffmpeg /usr/local/lib/ /usr/local/lib/
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /data
ENTRYPOINT ["timelapse"]
