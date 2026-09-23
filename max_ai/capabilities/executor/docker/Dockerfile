# Build from the repository root:
# docker build -f max_ai/capabilities/executor/docker/Dockerfile -t maxai-runtime:latest .
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.11-slim-bookworm

# Reuse the existing environment image's language and document utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils sed ripgrep ca-certificates curl git wget jq unzip zip \
    less tree fd-find sqlite3 openssl nodejs npm rustc cargo build-essential \
    pkg-config && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# Install the framework at build time so any importable CoreTool can run here.
COPY pyproject.toml README.md /opt/maxai/
COPY max_ai /opt/maxai/max_ai
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python /opt/maxai

ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN test "$AGENT_UID" -gt 0 && test "$AGENT_GID" -gt 0 \
    && groupadd --gid "$AGENT_GID" agent \
    && useradd --uid "$AGENT_UID" --gid "$AGENT_GID" --create-home agent \
    && mkdir /workspaces && chown agent:agent /workspaces
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    NPM_CONFIG_CACHE=/tmp/.cache/npm \
    CARGO_HOME=/tmp/.cargo
WORKDIR /workspaces
USER agent
CMD ["sleep", "infinity"]
