# Acme HR MCP server

An example MCP server (official MCP Python SDK 2.2) for Acme AI's HR system,
over streamable HTTP with JWT authentication.

- `main.py`: entry point.
- `src/_app.py`: the `MCPServer` instance and its authentication.
- `src/auth/jwt_handler.py`: creates and verifies tokens; `sub` is the employee ID.
- `src/tools/`: `get_vacation_balance`, `calculate_vacation_days`, `submit_vacation`, `send_email`, `set_parking`.
- `src/resources/`: `acme://guides/vacation` and `acme://guides/parking`, from `data/`.

From this directory:

```sh
cp .env.example .env            # set MCP_SEED to a long random string
uv sync
uv run python -m src.auth.jwt_handler --subject 0123456789 --hours 720   # your token
uv run main.py                  # http://127.0.0.1:8765/mcp
```

Connect from `max_ai` with the token in an environment variable:

```python
from max_ai.capabilities.mcp import HTTPServerConfig

HTTPServerConfig(server_id="acme_hr", url="http://127.0.0.1:8765/mcp", token_env="ACME_MCP_TOKEN")
```
