# The default runtime for DockerExecutor and ModalExecutor; both build it on first use.
# Bash, Python, Node and uv; the agent installs what else it needs. Nothing of max_ai.
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git jq unzip zip ripgrep \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN test "$AGENT_UID" -gt 0 && test "$AGENT_GID" -gt 0 \
    && groupadd --gid "$AGENT_GID" agent \
    && useradd --uid "$AGENT_UID" --gid "$AGENT_GID" --create-home agent \
    && mkdir /workspaces && chown agent:agent /workspaces \
    && chmod 755 /root
# The agent runs as AGENT_UID with HOME=/tmp: `pip install --user` and
# `npm install -g` write there and their commands land on PATH.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PATH=/tmp/.local/bin:/tmp/.npm-global/bin:$PATH \
    NPM_CONFIG_PREFIX=/tmp/.npm-global \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    NPM_CONFIG_CACHE=/tmp/.cache/npm
WORKDIR /workspaces
# No USER: executors drop to AGENT_UID themselves (Docker --user, Modal setpriv).
CMD ["sleep", "infinity"]
