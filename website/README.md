# Max AI website

Initial documentation site for Max AI.

The interface includes section navigation, dark mode, a responsive menu, and buttons to copy examples.

## Run the platform

From the repository root:

    python -m website.app

The site will be available at http://127.0.0.1:8000. You can also configure WEBSITE_HOST, WEBSITE_PORT, and WEBSITE_DEBUG.

The http://127.0.0.1:8000/health endpoint returns the server status.

The framework map is available at http://127.0.0.1:8000/architecture.md and in the repository root as `architecture.md`.

## Documentation pages

- `/` — overview and first working example
- `/agent.html` — Agent constructor and run lifecycle
- `/prompt-layers.html` — prompt stack, variable sources, rendering, and provider injection
- `/tools.html` — tools, dispatch, reasoning, and runtime services
- `/capabilities.html` — workspace, context registries, model providers, and completion
- `/integrations.html` — MCP, structured output, approvals, and application interfaces
- `/architecture.md` — framework package map and runtime architecture
