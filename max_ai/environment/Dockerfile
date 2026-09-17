FROM ghcr.io/astral-sh/uv:latest AS uv
FROM debian:bookworm-slim

# Runtime tooling belongs to the image, not the Python environment adapter.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils sed ripgrep ca-certificates curl git \
    wget jq unzip zip less tree fd-find sqlite3 openssl \
    python3 python3-venv nodejs npm rustc cargo build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# Match host workspace ownership at image build time when necessary.
ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN test "$AGENT_UID" -gt 0 && test "$AGENT_GID" -gt 0 \
    && groupadd --gid "$AGENT_GID" agent \
    && useradd --uid "$AGENT_UID" --gid "$AGENT_GID" --create-home agent \
    && mkdir /workspace && chown agent:agent /workspace

ENV WORKSPACE=/workspace \
    HOME=/workspace \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    CARGO_HOME=/workspace/.cargo \
    NPM_CONFIG_CACHE=/workspace/.cache/npm \
    NPM_CONFIG_PREFIX=/workspace/.npm
ENV PATH="/workspace/.npm/bin:/workspace/.cargo/bin:${PATH}"
WORKDIR /workspace
USER agent
CMD ["sleep", "infinity"]
