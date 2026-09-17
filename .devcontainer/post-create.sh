#!/usr/bin/env bash
set -euo pipefail

echo "Syncing Python dependencies..."
uv sync

echo "Starting LLM services..."
cd /max_ai/llm-models
docker compose --env-file .env up -d

echo "Pulling Ollama models..."
docker exec ollama ollama pull gemma4:e2b-it-q4_K_M
echo "Post-create setup complete."
  
# Install code 
export CODEX_NON_INTERACTIVE=1
if ! curl -fsSL https://chatgpt.com/codex/install.sh | bash; then
    echo "post-create.sh: Failed to install code, continue without it"
    exit 1
fi

if ! curl -fsSL https://claude.ai/install.sh | bash; then
    echo "post-create.sh: Failed to install claude, continue without it"
    exit 1
fi