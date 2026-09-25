"""Canonical folders and URLs for generated and hand-authored documentation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent.parent
SPECIAL_PAGES = {
    "index.html": "overview/index.html",
    "agent.html": "agents/index.html",
    "core.html": "core/index.html",
    "concepts.html": "core/concepts.html",
    "integrations.html": "core/integrations.html",
    "prompt-layers.html": "core/prompt-layers.html",
    "roadmap.html": "core/roadmap.html",
    "cli.html": "cli/index.html",
    "examples.html": "examples/index.html",
}

BASECOMPONENT_FILES = {
    "basecomponent.html", "clients.html", "compaction.html", "executor.html",
    "knowledge.html", "layers.html", "memory.html", "middleware.html",
    "reasoning.html", "skills.html", "tools.html", "workspace.html",
}

CAPABILITY_GROUPS = {
    "clients": "clients", "compaction": "compaction", "completion-gate": "completion-gates",
    "context": "context", "knowledge": "knowledge", "mcp": "mcp", "memory": "memory",
    "middleware": "middleware", "quota-store": "quota-stores", "reasoning": "reasoning",
    "session-store": "session-stores", "stack": "prompt-layers", "tool": "tools",
    "workspace": "workspaces", "skills": "skills", "executor": "executors",
    "output-format": "output-formats",
}


@lru_cache(maxsize=None)
def page_path(filename: str) -> str:
    """Return the physical path relative to the website root."""
    if filename in SPECIAL_PAGES:
        return SPECIAL_PAGES[filename]
    if filename.startswith("type-") and filename.endswith(".html"):
        return f"types/{filename[len('type-'):-len('.html')]}.html"
    if filename == "events.html":
        return "events/index.html"
    if filename == "base.html":
        return "basecomponents/index.html"
    if filename == "capabilities.html":
        return "capabilities/index.html"
    if filename.startswith("example-") and filename.endswith(".html"):
        slug = filename[len("example-"):-len(".html")]
        data = json.loads((WEBSITE / "data" / "example_catalog.json").read_text(encoding="utf-8"))
        entry = next((item for item in data if item["slug"] == slug), None)
        if entry:
            level = "basic" if entry["level"] == "Basic" else "advanced"
            return f"examples/{level}/{slug}.html"
    if filename in BASECOMPONENT_FILES:
        return f"basecomponents/{filename}"
    if filename.startswith("capability-") and filename.endswith(".html"):
        slug = filename[len("capability-"):-len(".html")]
        if slug == "output-format":
            return "capabilities/output-formats/structured.html"
        group, _, variant = slug.partition("-")
        if group in {"completion", "quota", "session", "output"}:
            # Preserve two-word category names while retaining the implementation variant.
            prefixes = {"completion-gate": "completion-gates", "quota-store": "quota-stores", "session-store": "session-stores", "output-format": "output-formats"}
            matched = next((key for key in prefixes if slug.startswith(key + "-")), None)
            if matched:
                return f"capabilities/{prefixes[matched]}/{slug[len(matched) + 1:]}.html"
        if slug.startswith("stack-"):
            return f"capabilities/prompt-layers/{slug[len('stack-'):]}.html"
        if slug.startswith("tool-"):
            tool = slug[len("tool-"):]
            if tool in {"bash", "ask-user", "filesystem", "plan"}:
                return f"capabilities/tools/native/{tool}.html"
            return f"capabilities/tools/decorator/{tool}.html"
        category = CAPABILITY_GROUPS.get(group)
        if category:
            return f"capabilities/{category}/{variant}.html"
    return filename


def page_url(filename: str) -> str:
    """Return an absolute site URL for a logical page name."""
    return "/" + page_path(filename)
