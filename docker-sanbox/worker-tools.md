# Docker Sandbox Image

This image is a plain bash/Python sandbox. It does not copy the MaxAI
framework and it does not copy tool repositories into the image.

Build the worker image from the repository root:

```bash
docker build -f docker-sanbox/Dockerfile.worker -t maxai-sandbox:py311 .
```

Runtime shape:

- `/server_workspace` is the mounted server workspace
- `$SESSIONS_DIR=/server_workspace/tmp/session`
- `$SKILLS_CACHE_DIR=/server_workspace/var/skills-cache`
- the default command is `/bin/bash`
- the image includes Python 3.11, `uv`, bash, curl, git, jq, ripgrep,
  unzip, and basic GNU utilities

Run it with the local server workspace mounted:

```bash
docker run --rm -it \
  -v "$PWD/server_workspace:/server_workspace" \
  maxai-sandbox:py311
```

Inside the container:

```bash
echo "$SESSIONS_DIR"
echo "$SKILLS_CACHE_DIR"
ls /server_workspace/tmp
ls /server_workspace/var
```

Those are defaults. Production can mount the same host workspace at a
different container path by configuring `DockerExecutor`:

```python
DockerExecutor(
    server_workspace="/srv/maxai/workspace",
    container_workspace="/runtime",
    sessions_subdir="sessions",
    skills_cache_subdir="cache/skills",
)
```

If a command needs another package or system tool, the LLM can install it
inside the running sandbox using normal shell commands.
