FROM python:3.14-slim

ARG YTDLP_VERSION=2026.07.04
ARG DENO_VERSION=2.9.2
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg unzip ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin yt-dlp for reproducible production builds.
RUN pip install --no-cache-dir uv \
    "yt-dlp[default]==${YTDLP_VERSION}"

# Pin Deno from a versioned upstream artifact so builds are reproducible.
RUN set -eux; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in \
        amd64) deno_arch='x86_64' ;; \
        arm64) deno_arch='aarch64' ;; \
        *) echo "unsupported architecture for Deno: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/deno.zip "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${deno_arch}-unknown-linux-gnu.zip"; \
    curl -fsSL -o /tmp/deno.zip.sha256 "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${deno_arch}-unknown-linux-gnu.zip.sha256sum"; \
    sha256="$(awk '{print $1}' /tmp/deno.zip.sha256)"; \
    echo "${sha256}  /tmp/deno.zip" | sha256sum -c -; \
    unzip -q /tmp/deno.zip -d /usr/local/bin; \
    chmod 0755 /usr/local/bin/deno; \
    rm -f /tmp/deno.zip /tmp/deno.zip.sha256

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["timeline-hub"]
CMD []
