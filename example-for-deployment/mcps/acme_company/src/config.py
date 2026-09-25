"""Settings read from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    MCP_NAME: str = os.getenv("MCP_NAME", "acme-hr")
    MCP_HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8765"))
    MCP_URL: str = os.getenv("MCP_URL", "http://127.0.0.1:8765")
    MCP_SEED: str = os.getenv("MCP_SEED", "")
    stateless_http: bool = True
    json_response: bool = True
    DATA_DIR: Path = PROJECT_ROOT / "data"


settings = Settings()
