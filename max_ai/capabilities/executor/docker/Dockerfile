# The default runtime for DockerExecutor and ModalExecutor; both build it on first use.
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils sed ripgrep ca-certificates curl git wget jq unzip zip \
    less tree fd-find sqlite3 openssl nodejs npm rustc cargo build-essential \
    pkg-config && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# The framework gets its own venv, called by absolute path by the executors;
# python/pip on PATH stay the agent's, which installs with `pip install --user`.
ARG MAXAI="maxai @ git+https://github.com/main-msfj/max_ai@main"
RUN uv venv /opt/maxai && uv pip install --python /opt/maxai/bin/python "$MAXAI"

ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN test "$AGENT_UID" -gt 0 && test "$AGENT_GID" -gt 0 \
    && groupadd --gid "$AGENT_GID" agent \
    && useradd --uid "$AGENT_UID" --gid "$AGENT_GID" --create-home agent \
    && mkdir /workspaces && chown agent:agent /workspaces \
    && chmod 755 /root
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    NPM_CONFIG_CACHE=/tmp/.cache/npm \
    CARGO_HOME=/tmp/.cargo
WORKDIR /workspaces
# No USER: executors drop to AGENT_UID themselves (Docker --user, Modal setpriv).
CMD ["sleep", "infinity"]
