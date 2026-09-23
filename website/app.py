"""Flask server for the Max AI documentation website."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

WEBSITE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = WEBSITE_DIR.parent

app = Flask(
    __name__,
    static_folder=str(WEBSITE_DIR),
    static_url_path="",
)


@app.get("/")
def index():
    """Serve the documentation home page."""
    return send_from_directory(WEBSITE_DIR, "index.html")


@app.get("/architecture.md")
def architecture():
    """Expose the repository architecture note from the documentation site."""
    return send_from_directory(REPOSITORY_DIR, "architecture.md", mimetype="text/markdown")


@app.get("/health")
def health():
    """Simple readiness endpoint for local tooling and deployments."""
    return jsonify({"status": "ok", "service": "max-ai-website"})


if __name__ == "__main__":
    app.run(
        host=os.getenv("WEBSITE_HOST", "127.0.0.1"),
        port=int(os.getenv("WEBSITE_PORT", "8000")),
        debug=os.getenv("WEBSITE_DEBUG", "false").lower() == "true",
    )
