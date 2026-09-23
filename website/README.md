# Max AI website

Documentation site for Max AI, with a shared layout across six topic pages.

The interface includes grouped navigation, an active page outline, local full-text
search (Ctrl/Cmd+K), light and dark themes, keyboard-accessible example tabs,
a mobile menu, and buttons that copy the displayed code.

The layout takes inspiration from the reading and navigation structure of
[Pydantic AI's documentation](https://pydantic.dev/docs/ai/overview/), with Max AI's
own branding and framework examples. It uses local HTML, CSS, JavaScript, and SVG;
no frontend build or external fonts are required.

## Run the platform

From the repository root:

    python -m website.app

The site will be available at http://127.0.0.1:8000. You can also configure WEBSITE_HOST, WEBSITE_PORT, and WEBSITE_DEBUG.

The http://127.0.0.1:8000/health endpoint returns the server status.

The framework map is available at http://127.0.0.1:8000/architecture.md and in the repository root as `architecture.md`.

## Update documentation

Edit the HTML pages directly. Shared styling and interactions live in
`styles.css` and `app.js`. Keep the navigation consistent across the six pages
and give each main section a stable `id` for links and the page outline.

After changing content, rebuild the checked-in search index:

    python website/build_search_index.py

The index builder uses only the Python standard library. Search runs in the
browser and loads `search-index.json` when the search dialog is first opened.
Code copy buttons read their adjacent code block, so no duplicate copy payload
needs updating.

## Documentation pages

- `/` — overview and first working example
- `/agent.html` — Agent constructor and run lifecycle
- `/prompt-layers.html` — prompt stack, variable sources, rendering, and provider injection
- `/tools.html` — tools, dispatch, reasoning, and runtime services
- `/capabilities.html` — workspace, context registries, model providers, and completion
- `/integrations.html` — MCP, structured output, approvals, and application interfaces
- `/architecture.md` — framework package map and runtime architecture
