"""Small Flask server for the AI harness research site."""

import os
from pathlib import Path

from flask import Flask, abort, redirect, send_from_directory

SITE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(SITE_DIR), static_url_path="")


@app.get("/")
def index():
    return redirect("/index.html")


@app.get("/<path:filename>")
def site_file(filename: str):
    # Only serve files that are inside this research site directory.
    requested = SITE_DIR / filename
    if not requested.is_file():
        abort(404)
    return send_from_directory(SITE_DIR, filename)


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
