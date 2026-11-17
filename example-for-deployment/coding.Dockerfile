# The coding agent's sandbox: python, pypdf (read_file uses it for PDFs) and git.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pypdf \
    && git config --system user.name "Coder" && git config --system user.email "coder@sandbox" \
    && git config --system init.defaultBranch main
