#!/usr/bin/env bash
set -e

echo "Starting LLM services..."

cd /max_ai/llm-models
docker compose --env-file .env up -d