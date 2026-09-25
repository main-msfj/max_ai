"""Keep the sidebar tree consistent across the static documentation pages."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from site_paths import page_path, page_url

HERE = Path(__file__).resolve().parent.parent
PAGES = (
    "index.html",
    "agent.html",
    "basecomponent.html",
    "clients.html",
    "compaction.html",
    "executor.html",
    "knowledge.html",
    "layers.html",
    "memory.html",
    "middleware.html",
    "reasoning.html",
    "skills.html",
    "tools.html",
    "workspace.html",
    "capability-clients-ollama.html", "capability-clients-openai.html", "capability-clients-openrouter.html",
    "capability-compaction-summary.html", "capability-compaction-sliding-window.html",
    "capability-completion-gate-runtime.html",
    "capability-context-local.html", "capability-context-sqlite.html",
    "capability-knowledge-local.html", "capability-knowledge-sqlite.html", "capability-knowledge-mongodb.html",
    "capability-mcp-stdio.html", "capability-mcp-http.html",
    "capability-memory-local.html", "capability-memory-sqlite.html", "capability-memory-mongodb.html",
    "capability-middleware-budget.html", "capability-middleware-logging.html", "capability-middleware-tracing.html",
    "capability-quota-store-local.html", "capability-quota-store-mongodb.html",
    "capability-reasoning-react.html", "capability-session-store-local.html",
    "capability-stack-agent-policy.html", "capability-stack-context.html", "capability-stack-knowledge.html",
    "capability-stack-memory.html", "capability-stack-rendering.html", "capability-stack-session-state.html",
    "capability-stack-skills.html", "capability-stack-task-analysis.html",
    "capability-tool-decorator.html", "capability-tool-function.html", "capability-tool-bash.html",
    "capability-tool-ask-user.html", "capability-tool-filesystem.html", "capability-tool-plan.html",
    "capability-workspace-local.html", "capability-workspace-azure-blob.html", "capability-workspace-minio.html",
    "capability-skills-local.html", "capability-skills-github.html",
    "capability-executor-local.html", "capability-executor-docker.html", "capability-executor-modal.html",
    "capability-output-format.html",
    "base.html",
    "core.html",
    "capabilities.html",
    "prompt-layers.html",
    "integrations.html",
    "examples.html",
    "concepts.html",
    "roadmap.html",
    "type-agent-response.html",
    "type-completion.html",
    "type-tool.html",
    "type-tool-call.html",
    "type-run-context.html",
    "events.html",
    "cli.html",
)

# A third value marks a child link in the visual tree.
BASIC_EXAMPLES = (
    ("Basic agent", "basic-agent"),
    ("Ollama agent", "ollama-agent"),
    ("Function tools", "function-tools"),
    ("Tool approval", "tool-approval"),
    ("Sessions", "sessions"),
    ("Memory", "memory"),
    ("Knowledge search", "knowledge-search"),
    ("Structured output", "structured-output"),
    ("MCP tools", "mcp-tools"),
    ("Skills", "skills"),
    ("Streaming events", "streaming-events"),
    ("Middleware and tracing", "middleware-tracing"),
    ("Workspace files", "workspace-files"),
    ("Concurrent runs", "concurrent-runs"),
    ("Full agent", "full-agent"),
)
ADVANCED_EXAMPLES = (
    ("Native tool", "native-tool"),
    ("PostgreSQL memory", "postgres-memory"),
    ("PostgreSQL knowledge", "postgres-knowledge"),
    ("Custom middleware", "custom-middleware"),
    ("In-memory session store", "in-memory-session-store"),
    ("Custom prompt layer", "custom-prompt-layer"),
    ("Custom embedding", "custom-embedding"),
    ("Custom workspace", "custom-workspace"),
    ("Serialize an agent", "serialize-agent"),
    ("Deserialize an agent", "deserialize-agent"),
)
PAGES += tuple(f"example-{slug}.html" for _, slug in (*BASIC_EXAMPLES, *ADVANCED_EXAMPLES))
PAGES = tuple(page_path(page) for page in PAGES)

MEMORY_BACKENDS = (
    ("Local", "memory-local"),
    ("SQLite", "memory-sqlite"),
    ("MongoDB", "memory-mongodb"),
)
KNOWLEDGE_BACKENDS = (
    ("Local", "knowledge-local"),
    ("SQLite", "knowledge-sqlite"),
    ("MongoDB", "knowledge-mongodb"),
)
GROUPS = (
    ("Start", (("Overview", "/overview/index.html", False),)),
    ("Agents", (("Agent", "/agents/index.html#agent-top", False),)),
    ("BaseComponents", (
        ("Clients", "/basecomponents/clients.html#clients-top", True),
        ("Compaction", "/basecomponents/compaction.html#compaction-top", True),
        ("Executor", "/basecomponents/executor.html#executor-top", True),
        ("Knowledge", "/basecomponents/knowledge.html#knowledge-top", True),
        ("Layers", "/basecomponents/layers.html#layers-top", True),
        ("Memory", "/basecomponents/memory.html#memory-top", True),
        ("Middleware", "/basecomponents/middleware.html#middleware-top", True),
        ("Reasoning", "/basecomponents/reasoning.html#reasoning-top", True),
        ("Skills", "/basecomponents/skills.html#skills-top", True),
        ("Tools", "/basecomponents/tools.html#tools-top", True),
        ("Workspace", "/basecomponents/workspace.html#workspace-top", True),
        ("More in GitHub", "https://github.com/main-msfj/max_ai/tree/main/max_ai/base", True),
    )),
    ("Capabilities", ()),
    ("Examples", ()),
    ("Types", (
        ("AgentResponse", "/types/agent-response.html#agent-response-top", True),
        ("Completion", "/types/completion.html#completion-top", True),
        ("Tool", "/types/tool.html#tool-top", True),
        ("Tool call", "/types/tool-call.html#tool-call-top", True),
        ("RunContext", "/types/run-context.html#run-context-top", True),
    )),
    ("Events", (("Event types", "/events/index.html#events-top", True),)),
    ("CLI", (("Command line", "/cli/index.html#cli-top", True),)),
)


def sidebar(page: str) -> str:
    chunks = ['<nav aria-label="Documentation">']
    active_assigned = False
    for title, links in GROUPS:
        chunks.append('<div class="nav-group nav-group-start">' if title == "Start" else '<div class="nav-group">')
        if title not in {"Examples", "CLI", "Start", "Agents", "BaseComponents", "Capabilities", "Types", "Events"}:
            chunks.append(f'<p class="nav-label">{escape(title)}</p>')
        if title == "BaseComponents":
            base_pages = {href.split("#", 1)[0].lstrip("/") for _, href, _ in links if not href.startswith("https:")}
            opened = page in base_pages or page == "basecomponents/basecomponent.html"
            active_attr = ' aria-current="location"' if opened else ""
            open_attr = " open" if opened else ""
            chunks.append(f'<details class="nav-disclosure nav-basecomponents"{open_attr}>')
            chunks.append(f'<summary{active_attr} class="nav-label nav-label-basecomponents">BaseComponents</summary>')
            chunks.append('<div class="nav-children">')
            for label, href, child in links:
                route = href.split("#", 1)[0]
                active = not active_assigned and route == "/" + page
                if active:
                    active_assigned = True
                classes = "nav-link" + (" nav-sub-link" if child else "") + (" active" if active else "")
                current = ' aria-current="page"' if active else ""
                target = ' target="_blank" rel="noopener"' if href.startswith("https://") else ""
                chunks.append(f'<a{current} class="{classes}" href="{href}"{target}>{escape(label)}</a>')
            chunks.append('</div></details>')
            chunks.append('</div>')
            continue
        if title in {"Start", "Agents"}:
            label = title
            opened = page == page_path("index.html") or (title == "Agents" and page == page_path("agent.html"))
            current = ' aria-current="location"' if opened else ""
            open_attr = " open" if opened else ""
            chunks.append(f'<details class="nav-disclosure nav-section"{open_attr}>')
            chunks.append(f'<summary{current} class="nav-label nav-label-section">{label}</summary>')
            chunks.append('<div class="nav-children">')
            for label, href, _ in links:
                route = href.split("#", 1)[0]
                active = route == "/" + page
                item_current = ' aria-current="page"' if active else ""
                chunks.append(f'<a{item_current} class="nav-link nav-sub-link{" active" if active else ""}" href="{href}">{escape(label)}</a>')
            chunks.append('</div></details>')
            chunks.append('</div>')
            continue
        if title in {"Types", "Events", "CLI"}:
            nav_class = {"Types": "nav-types", "Events": "nav-events", "CLI": "nav-cli"}[title]
            route_for_page = "/" + page
            opened = any(href.split("#", 1)[0] == route_for_page for _, href, _ in links)
            open_attr = " open" if opened else ""
            current = ' aria-current="location"' if opened else ""
            chunks.append(f'<details class="nav-disclosure {nav_class}"{open_attr}>')
            chunks.append(f'<summary{current} class="nav-label nav-label-capabilities">{title}</summary>')
            chunks.append('<div class="nav-children">')
            for label, href, _ in links:
                active = href.split("#", 1)[0] == route_for_page
                item_current = ' aria-current="page"' if active else ""
                item_class = " active" if active else ""
                chunks.append(f'<a{item_current} class="nav-link nav-sub-link{item_class}" href="{href}">{escape(label)}</a>')
            chunks.append('</div></details>')
            chunks.append('</div>')
            continue

        for label, href, child in links:
            route = href.split("#", 1)[0]
            active = not active_assigned and route == "/" + page
            if active:
                active_assigned = True
            classes = "nav-link" + (" nav-sub-link" if child else "") + (" active" if active else "")
            current = ' aria-current="location"' if active else ""
            target = ' target="_blank" rel="noopener"' if href.startswith("https://") else ""
            chunks.append(f'<a{current} class="{classes}" href="{href}"{target}>{escape(label)}</a>')

        if title == "Capabilities":
            capability_page = page.startswith("capabilities/")
            capability_overview = page == "capabilities/index.html"
            cap_open_attr = " open" if capability_page or capability_overview else ""
            cap_current = ' aria-current="location"' if capability_page or capability_overview else ""
            chunks.append(f'<details class="nav-disclosure nav-capability-root"{cap_open_attr}>')
            chunks.append(f'<summary{cap_current} class="nav-label nav-label-capabilities">Capabilities</summary>')
            chunks.append('<div class="nav-children">')
            capability_groups = (
                ("Client", (("Ollama", "capability-clients-ollama.html"), ("OpenAI", "capability-clients-openai.html"), ("OpenRouter", "capability-clients-openrouter.html"))),
                (
                    "Tool",
                    (
                        ("Native", (("Bash", "capability-tool-bash.html"), ("Ask User", "capability-tool-ask-user.html"), ("File system", "capability-tool-filesystem.html"), ("Plan", "capability-tool-plan.html"))),
                        ("Decorator", (("Decorator", "capability-tool-decorator.html"), ("Function wrapper", "capability-tool-function.html"))),
                    ),
                ),
                ("MCP", (("Stdio", "capability-mcp-stdio.html"), ("HTTP", "capability-mcp-http.html"))),
                ("Memory", (("Local", "capability-memory-local.html"), ("SQLite", "capability-memory-sqlite.html"), ("MongoDB", "capability-memory-mongodb.html"))),
                ("Knowledge", (("Local", "capability-knowledge-local.html"), ("SQLite", "capability-knowledge-sqlite.html"), ("MongoDB", "capability-knowledge-mongodb.html"))),
                ("Skills", (("Local", "capability-skills-local.html"), ("GitHub", "capability-skills-github.html"))),
                ("Reasoning", (("ReAct", "capability-reasoning-react.html"),)),
                ("Compaction", (("Summary", "capability-compaction-summary.html"), ("Sliding window", "capability-compaction-sliding-window.html"))),
                ("Middleware", (("Budget", "capability-middleware-budget.html"), ("Logging", "capability-middleware-logging.html"), ("Tracing", "capability-middleware-tracing.html"))),
                ("Gates", (("Runtime completion gate", "capability-completion-gate-runtime.html"),)),
                ("Output format", (("Structured output", "capability-output-format.html"),)),
                ("Prompt layers", (("Agent policy", "capability-stack-agent-policy.html"), ("Context", "capability-stack-context.html"), ("Knowledge", "capability-stack-knowledge.html"), ("Memory", "capability-stack-memory.html"), ("Rendering", "capability-stack-rendering.html"), ("Session state", "capability-stack-session-state.html"), ("Skills", "capability-stack-skills.html"), ("Task analysis", "capability-stack-task-analysis.html"))),
                ("Workspace", (("Local", "capability-workspace-local.html"), ("Azure Blob", "capability-workspace-azure-blob.html"), ("MinIO", "capability-workspace-minio.html"))),
                ("Executor", (("Local", "capability-executor-local.html"), ("Docker", "capability-executor-docker.html"), ("Modal", "capability-executor-modal.html"))),
                ("Context", (("Local", "capability-context-local.html"), ("SQLite", "capability-context-sqlite.html"))),
                ("Session store", (("Local", "capability-session-store-local.html"),)),
                ("Quota store", (("Local", "capability-quota-store-local.html"), ("MongoDB", "capability-quota-store-mongodb.html"))),
            )
            for category, raw_items in capability_groups:
                nested = category == "Tool"
                if nested:
                    item_pages = {page_path(filename) for _, subitems in raw_items for _, filename in subitems}
                else:
                    item_pages = {page_path(filename) for _, filename in raw_items}
                opened = page in item_pages
                active = opened
                open_attr = " open" if opened else ""
                active_attr = ' aria-current="location"' if active else ""
                active_class = " active" if active else ""
                chunks.append(f'<details class="nav-disclosure nav-capability"{open_attr}>')
                chunks.append(f'<summary{active_attr} class="nav-link{active_class}">{category}</summary>')
                chunks.append('<div class="nav-children">')
                if nested:
                    for sublabel, subitems in raw_items:
                        sub_open = " open" if page in {page_path(filename) for _, filename in subitems} else ""
                        chunks.append(f'<details class="nav-disclosure nav-category"{sub_open}><summary class="nav-link nav-sub-link">{sublabel}</summary><div class="nav-children">')
                        for item, filename in subitems:
                            item_active = page == page_path(filename)
                            item_current = ' aria-current="page"' if item_active else ""
                            item_class = " active" if item_active else ""
                            chunks.append(f'<a{item_current} class="nav-link nav-example-link{item_class}" href="{page_url(filename)}#{filename.removesuffix(".html")}-top">{item}</a>')
                        chunks.append('</div></details>')
                else:
                    for item, filename in raw_items:
                        item_active = page == page_path(filename)
                        item_current = ' aria-current="page"' if item_active else ""
                        item_class = " active" if item_active else ""
                        link_id = f"{filename.removesuffix('.html')}-top"
                        chunks.append(f'<a{item_current} class="nav-link nav-sub-link{item_class}" href="{page_url(filename)}#{link_id}">{item}</a>')
                chunks.append('</div></details>')
            chunks.append('</div></details>')

        if title == "Examples":
            example_pages = {page_path(f"example-{slug}.html") for _, slug in (*BASIC_EXAMPLES, *ADVANCED_EXAMPLES)}
            opened = page in example_pages or page == page_path("examples.html")
            active_attr = ' aria-current="location"' if opened else ""
            active_class = " active" if opened else ""
            open_attr = " open" if opened else ""
            chunks.append(f'<details class="nav-disclosure nav-examples"{open_attr}>')
            examples_class = "nav-label nav-label-capabilities" + (" active" if opened else "")
            chunks.append(f'<summary{active_attr} class="{examples_class}">Examples</summary>')
            chunks.append('<div class="nav-children">')
            for category, items in (("Basic", BASIC_EXAMPLES), ("Advanced", ADVANCED_EXAMPLES)):
                category_open = " open" if page in {page_path(f"example-{slug}.html") for _, slug in items} else ""
                chunks.append(f'<details class="nav-disclosure nav-category"{category_open}>')
                chunks.append(f'<summary class="nav-link nav-sub-link">{category}</summary>')
                chunks.append('<div class="nav-children">')
                for item, anchor in items:
                    current = ' aria-current="page"' if page == page_path(f"example-{anchor}.html") else ""
                    selected = " active" if current else ""
                    chunks.append(f'<a{current} class="nav-link nav-example-link{selected}" href="{page_url(f"example-{anchor}.html")}#{anchor}">{escape(item)}</a>')
                chunks.append('</div></details>')
            chunks.append('</div></details>')
        chunks.append('</div>')
    chunks.append('</nav>')
    return "\n".join(chunks)


def normalize_toc_links(source: str) -> str:
    """Give every page outline link the shared block spacing and active style."""
    pattern = re.compile(
        r'(<(?:details|aside)\b[^>]*class="[^"]*\b(?:mobile-toc|page-toc)\b[^"]*"[^>]*>'
        r'.*?<nav\b[^>]*>)(.*?)(</nav>)',
        re.DOTALL,
    )

    def add_class(match: re.Match[str]) -> str:
        body = re.sub(r'<a(?![^>]*\bclass=)', '<a class="toc-link"', match.group(2))
        return match.group(1) + body + match.group(3)

    return pattern.sub(add_class, source)



def main() -> None:
    pattern = re.compile(r'<nav aria-label="Documentation">.*?</nav>', re.DOTALL)
    for name in PAGES:
        path = HERE / name
        source = path.read_text(encoding="utf-8")
        for _, slug in (*BASIC_EXAMPLES, *ADVANCED_EXAMPLES):
            source = source.replace(f'href="/examples/index.html#{slug}"', f'href="{page_url(f"example-{slug}.html")}#{slug}"')
        updated, count = pattern.subn(sidebar(name), source, count=1)
        if count != 1:
            raise ValueError(f"No sidebar navigation in {path}")
        updated = re.sub(
            r'(<body\b[^>]*\bdata-page=")[^"]+("[^>]*>)',
            lambda match: match.group(1) + name + match.group(2),
            updated,
            count=1,
        )
        path.write_text(updated, encoding="utf-8")
    # Hand-authored and generated pages use the same TOC link presentation.
    for path in HERE.rglob("*.html"):
        source = path.read_text(encoding="utf-8")
        updated = normalize_toc_links(source)
        if updated != source:
            path.write_text(updated, encoding="utf-8")
    print(f"Updated navigation in {len(PAGES)} pages.")


if __name__ == "__main__":
    main()
