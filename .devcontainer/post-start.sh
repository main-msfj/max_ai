#!/usr/bin/env bash
set -e

# Start the infrastructure (Ollama, MongoDB, MinIO, Azurite, Langfuse). Compose
# only creates or starts what is missing, so running containers are untouched.
# Never block the devcontainer: if Docker fails, the log says why.
echo "Starting docker-infra..."
(cd /max_ai/docker-infra && docker compose --profile observability up -d) \
    >> /tmp/max-ai-docker-infra.log 2>&1 \
    || echo "docker-infra did not start; see /tmp/max-ai-docker-infra.log"

# The infrastructure UIs run in other containers. Rejoin their networks after
# a devcontainer recreation and expose them on this container's loopback so
# VS Code can forward ordinary numeric ports.
for network in max-ai-infra_capabilities max-ai-infra_observability max-ai-infra_llm; do
    if docker network inspect "$network" >/dev/null 2>&1; then
        docker network connect "$network" "$(hostname)" 2>/dev/null || true
    fi
done
pkill -f infra_port_forward.py 2>/dev/null || true
nohup python3 /max_ai/.devcontainer/infra_port_forward.py \
    >> /tmp/max-ai-infra-port-forward.log 2>&1 < /dev/null &
