#!/usr/bin/env bash
set -e

echo "Starting LLM services..."

cd /max_ai/llm-models
docker compose --env-file .env up -d

# The infrastructure UIs run in other containers. Rejoin their networks after
# a devcontainer recreation and expose them on this container's loopback so
# VS Code can forward ordinary numeric ports.
for network in max-ai-infra_backend max-ai-infra_observability; do
    if docker network inspect "$network" >/dev/null 2>&1; then
        docker network connect "$network" "$(hostname)" 2>/dev/null || true
    fi
done
nohup python3 /max_ai/.devcontainer/infra_port_forward.py \
    >> /tmp/max-ai-infra-port-forward.log 2>&1 < /dev/null &
