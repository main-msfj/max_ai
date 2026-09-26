#!/usr/bin/env bash
# Start the Acme HR MCP server from any directory: http://127.0.0.1:8765/mcp
cd "$(dirname "$0")"
exec ../../../.venv/bin/python main.py
