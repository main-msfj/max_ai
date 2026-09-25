"""Generate concise reference pages for the concrete capabilities."""

from __future__ import annotations

import ast
import re
import textwrap
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
from site_paths import page_path

# filename, category, title, purpose, implementation, config, config_file, config_note
PAGES = [
    ("capability-clients-ollama.html", "Clients", "Ollama client", "Connects an agent to a model served by Ollama. It uses the OpenAI-compatible chat API and supports Ollama-specific thinking and model options.", "max_ai/capabilities/clients/ollama/client.py · OllamaChatCompletionClient", "OllamaChatCompletionClientConfig", "max_ai/capabilities/clients/ollama/_model.py", "Selects the model and host, and carries API key environment, thinking, keep-alive, and request options."),
    ("capability-clients-openai.html", "Clients", "OpenAI client", "Sends chat requests to OpenAI and adapts provider responses to the framework client contract.", "max_ai/capabilities/clients/openai/client.py · OpenAIChatCompletionClient", "OpenAIChatCompletionClientConfig", "max_ai/capabilities/clients/openai/_model.py", "Defines the model, API key environment, optional base URL, organization, project, and request options."),
    ("capability-clients-openrouter.html", "Clients", "OpenRouter client", "Uses OpenRouter's OpenAI-compatible API to route requests to supported models and providers.", "max_ai/capabilities/clients/openrouter/client.py · OpenRouterChatCompletionClient", "OpenRouterChatCompletionClientConfig", "max_ai/capabilities/clients/openrouter/_model.py", "Defines model routing, fallback models, provider and reasoning preferences, app metadata, and request options."),
    ("capability-compaction-summary.html", "Compaction", "Summary compaction", "Replaces older conversation history with a summary when the run needs a smaller prompt.", "max_ai/capabilities/compaction/summary/_strategy.py · SummaryCompaction", "SummaryCompactionConfig", "max_ai/capabilities/compaction/summary/_model.py", "Inherits common compaction settings and sets summary and message token limits."),
    ("capability-compaction-sliding-window.html", "Compaction", "Sliding window compaction", "Keeps a configured number of recent turns and drops older turns from the active context.", "max_ai/capabilities/compaction/window/_strategy.py · SlidingWindowCompaction", "SlidingWindowCompactionConfig", "max_ai/capabilities/compaction/window/_model.py", "Inherits common compaction settings and defines the maximum number of turns to keep."),
    ("capability-completion-gate-runtime.html", "Completion gate", "Runtime completion gate", "Checks runtime conditions before an agent is allowed to finish a run, such as whether a plan is complete or a command failed.", "max_ai/capabilities/completion_gate/gate.py · RuntimeCompletionGate", "RuntimeGateConfig", "max_ai/capabilities/completion_gate/_model.py", "Controls whether the gate is enabled and which plan and command checks it applies."),
    ("capability-context-local.html", "Context", "Local context registry", "Stores and retrieves run context in a local filesystem location, scoped by user and session.", "max_ai/capabilities/context/local/_registry.py · LocalContextRegistry", "LocalContextRegistryConfig", "max_ai/capabilities/context/local/_model.py", "Sets user and session identifiers, storage path, and tool mode."),
    ("capability-context-sqlite.html", "Context", "SQLite context registry", "Persists run context in SQLite for local applications that need context to survive process restarts.", "max_ai/capabilities/context/sqlite/_registry.py · SQLiteContextRegistry", "SQLiteContextRegistryConfig", "max_ai/capabilities/context/sqlite/_model.py", "Sets user and session identifiers, database location and name, and tool mode."),
    ("capability-knowledge-local.html", "Knowledge", "Local knowledge registry", "Searches knowledge documents stored in a local directory and returns relevant material to the agent.", "max_ai/capabilities/knowledge/local/_registry.py · LocalKnowledgeRegistry", "LocalKnowledgeRegistryConfig", "max_ai/capabilities/knowledge/local/_model.py", "Defines the source name and description, document directory, tool mode, and optional embeddings."),
    ("capability-knowledge-sqlite.html", "Knowledge", "SQLite knowledge registry", "Indexes and searches knowledge documents with a SQLite-backed store.", "max_ai/capabilities/knowledge/sqlite/_registry.py · SQLiteKnowledgeRegistry", "SQLiteKnowledgeRegistryConfig", "max_ai/capabilities/knowledge/sqlite/_model.py", "Defines source metadata, storage path and database name, minimum score, tool mode, and optional embeddings."),
    ("capability-knowledge-mongodb.html", "Knowledge", "MongoDB knowledge registry", "Stores and searches knowledge records in MongoDB, including vector search when embeddings are configured.", "max_ai/capabilities/knowledge/mongodb/_registry.py · MongoDBKnowledgeRegistry", "MongoDBKnowledgeRegistryConfig", "max_ai/capabilities/knowledge/mongodb/_model.py", "Defines source metadata, database and collection, URI environment variable, search threshold, and optional embeddings."),
    ("capability-mcp-stdio.html", "MCP", "MCP stdio transport", "Starts an MCP server as a local subprocess and exposes its tools to the agent.", "max_ai/capabilities/mcp/client_manager.py · MCPClientManager; max_ai/capabilities/mcp/integration.py · create_mcp_tools", "StdioMCPServerConfig", "max_ai/capabilities/mcp/_model.py", "Extends MCPServerConfig with command, arguments, environment, and approval settings."),
    ("capability-mcp-http.html", "MCP", "MCP HTTP transport", "Connects to a remote MCP server over HTTP and exposes its tools to the agent.", "max_ai/capabilities/mcp/client_manager.py · MCPClientManager; max_ai/capabilities/mcp/integration.py · create_mcp_tools", "HTTPServerConfig", "max_ai/capabilities/mcp/_model.py", "Extends MCPServerConfig with URL, headers, environment-based credentials, and approval settings."),
    ("capability-memory-local.html", "Memory", "Local memory registry", "Stores user or session memories on disk and retrieves relevant entries in later runs.", "max_ai/capabilities/memory/local/_registry.py · LocalMemoryRegistry", "LocalMemoryRegistryConfig", "max_ai/capabilities/memory/local/_model.py", "Sets storage path, user and session scope, context age, tool mode, and optional embeddings."),
    ("capability-memory-sqlite.html", "Memory", "SQLite memory registry", "Persists searchable user or session memories in a local SQLite database.", "max_ai/capabilities/memory/sqlite/_registry.py · SQLiteMemoryRegistry", "SQLiteMemoryRegistryConfig", "max_ai/capabilities/memory/sqlite/_model.py", "Sets database location and name, user and session scope, context age, result limit, and optional embeddings."),
    ("capability-memory-mongodb.html", "Memory", "MongoDB memory registry", "Persists searchable user or session memories in MongoDB for shared or deployed applications.", "max_ai/capabilities/memory/mongodb/_registry.py · MongoDBMemoryRegistry", "MongoDBMemoryRegistryConfig", "max_ai/capabilities/memory/mongodb/_model.py", "Sets database and collection, URI environment variable, user and session scope, context age, and search options."),
    ("capability-middleware-budget.html", "Middleware", "Budget middleware", "Tracks run usage and applies configured limits for time, model calls, tool calls, tokens, or cost.", "max_ai/capabilities/middleware/budget.py · BudgetMiddleware", "BudgetConfig", "max_ai/capabilities/middleware/budget.py", "Defines usage limits and can use a quota and quota store for persistent accounting."),
    ("capability-middleware-logging.html", "Middleware", "Logging middleware", "Records selected agent lifecycle activity through the framework middleware hooks.", "max_ai/capabilities/middleware/logging.py · LoggingMiddleware", "LoggingMiddlewareConfig", "max_ai/capabilities/middleware/logging.py", "Controls logging behavior for the middleware."),
    ("capability-middleware-tracing.html", "Middleware", "Tracing middleware", "Adds tracing around agent execution so runs and their operations can be inspected.", "max_ai/capabilities/middleware/tracing.py · TracingMiddleware", "TracingConfig", "max_ai/capabilities/middleware/tracing.py", "Carries the tracing middleware options and instrumentation settings."),
    ("capability-quota-store-local.html", "Quota store", "Local quota store", "Persists quota usage locally so limits can be counted across runs.", "max_ai/capabilities/quota_store/local/_store.py · LocalQuotaStore", "LocalQuotaStoreConfig", "max_ai/capabilities/quota_store/local/_model.py", "Defines the local data path and how many quota periods to retain."),
    ("capability-quota-store-mongodb.html", "Quota store", "MongoDB quota store", "Stores quota usage in MongoDB so multiple processes can share accounting data.", "max_ai/capabilities/quota_store/mongodb/_store.py · MongoDBQuotaStore", "MongoDBQuotaStoreConfig", "max_ai/capabilities/quota_store/mongodb/_model.py", "Defines the database, collection, URI environment variable, and connection timeout."),
    ("capability-reasoning-react.html", "Reasoning", "ReAct reasoning loop", "Runs the agent's repeated model, tool, and completion steps using the ReAct strategy.", "max_ai/capabilities/reasoning/react/loop.py · ReactLoop", "ReactLoopConfig", "max_ai/capabilities/reasoning/react/loop.py", "Sets the maximum loop iterations and the configured loop guards."),
    ("capability-session-store-local.html", "Session store", "Local session store", "Saves and loads serialized session state from local storage.", "max_ai/capabilities/session_store/local/_store.py · LocalSessionStore", "LocalSessionStoreConfig", "max_ai/capabilities/session_store/local/_model.py", "Defines the local base path used for session data."),
    ("capability-stack-agent-policy.html", "Stacks", "Agent policy layer", "Adds policy and behavior guidance to the composed agent prompt.", "max_ai/capabilities/stacks/agent_policy_layer.py · AgentPolicyLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-context.html", "Stacks", "Context layer", "Renders run context into the composed prompt for the agent.", "max_ai/capabilities/stacks/context_layer.py · ContextLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-knowledge.html", "Stacks", "Knowledge layer", "Adds retrieved knowledge to the composed prompt when the agent has relevant sources.", "max_ai/capabilities/stacks/knowledge_layer.py · KnowledgeLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-memory.html", "Stacks", "Memory layer", "Renders relevant user or session memories into the composed prompt.", "max_ai/capabilities/stacks/memory_layer.py · MemoryLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-rendering.html", "Stacks", "Rendering layer", "Applies a prompt template and variables to render a layer in the agent's prompt.", "max_ai/capabilities/stacks/rendering_layer.py · RenderingLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-session-state.html", "Stacks", "Session state layer", "Makes selected session state available as a layer in the composed prompt.", "max_ai/capabilities/stacks/session_state_layer.py · SessionStateLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-skills.html", "Stacks", "Skills layer", "Adds selected skill instructions and related context to the composed prompt.", "max_ai/capabilities/stacks/skills_layer.py · SkillsLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-stack-task-analysis.html", "Stacks", "Task analysis layer", "Adds task analysis guidance to the composed prompt before execution.", "max_ai/capabilities/stacks/task_analysis_layer.py · TaskAnalysisLayer", "StackConfig", "max_ai/core/model/stacks.py", "The shared stack config carries the layer name, instructions, description, template, and related metadata."),
    ("capability-tool-decorator.html", "Tools", "Tool decorator", "Turns a Python function into a tool by describing its purpose and parameters through a decorator.", "max_ai/capabilities/tools/decorator.py · tool", None, None, "The decorator derives the tool schema from the function and its type hints; there is no dedicated config model."),
    ("capability-tool-function.html", "Tools", "Function tool wrapper", "Wraps a Python callable as a framework tool and derives its input schema from its signature.", "max_ai/capabilities/tools/function_as_tool.py · FunctionAsTool", None, None, "There is no dedicated config model; the callable signature and docstring describe the tool."),
    ("capability-tool-bash.html", "Tools", "Bash tool", "Lets an agent request shell commands through the configured execution environment and permissions.", "max_ai/capabilities/tools/bash/_tool.py · BashTool", "BashPermissions", "max_ai/capabilities/tools/bash/_permissions.py", "This permissions model controls allowed command behavior; it is not a component *Config model."),
    ("capability-tool-ask-user.html", "Tools", "Ask user tool", "Pauses a run to collect information or approval from the user through the runtime interaction flow.", "max_ai/capabilities/tools/ask_user/_tool.py · AskUserTool", None, None, "There is no dedicated config model; input and approval are handled through the tool's runtime interaction."),
    ("capability-tool-filesystem.html", "Tools", "File system tools", "Provides tools for reading and changing files in the agent's configured workspace.", "max_ai/capabilities/tools/file_system/_toolset.py · FileSystemTools", None, None, "There is no dedicated config model; workspace configuration controls the backing file store."),
    ("capability-tool-plan.html", "Tools", "Plan tool", "Lets an agent create and update a structured plan during a run.", "max_ai/capabilities/tools/plan/_agent_tool.py · AgentUpdatePlanTool", None, None, "There is no dedicated config model. AgentPlan and PlanStep are runtime data models."),
    ("capability-workspace-local.html", "Workspace", "Local workspace", "Stores user files in a local directory and exposes workspace operations to the agent.", "max_ai/capabilities/workspace/local/_workspace.py · LocalWorkspace", "LocalWorkspaceConfig", "max_ai/capabilities/workspace/local/_model.py", "Extends WorkspaceConfig with local workspace settings."),
    ("capability-workspace-azure-blob.html", "Workspace", "Azure Blob workspace", "Stores workspace files in Azure Blob Storage and uses a local cache for file operations.", "max_ai/capabilities/workspace/azure_blob/_workspace.py · AzureBlobWorkspace", "AzureBlobWorkspaceConfig", "max_ai/capabilities/workspace/azure_blob/_model.py", "Extends WorkspaceConfig with blob URL, credential environment variable, and cache settings."),
    ("capability-workspace-minio.html", "Workspace", "MinIO workspace", "Stores workspace files in an S3-compatible MinIO bucket and uses a local cache for file operations.", "max_ai/capabilities/workspace/minio/_workspace.py · MinIOWorkspace", "MinIOWorkspaceConfig", "max_ai/capabilities/workspace/minio/_model.py", "Extends WorkspaceConfig with endpoint, bucket, credential environment variables, region, and cache settings."),
    ("capability-skills-local.html", "Skills", "Local skill registry", "Loads task instructions and supporting files from a local skills directory for the agent to consult.", "max_ai/capabilities/skills/local/_registry.py · LocalSkillRegistry", "LocalSkillRegistryConfig", "max_ai/capabilities/skills/local/_model.py", "Defines the local skill directory and registry options."),
    ("capability-skills-github.html", "Skills", "GitHub skill registry", "Loads skill instructions and supporting files from a GitHub repository.", "max_ai/capabilities/skills/github/_registry.py · GithubSkillRegistry", "GithubSkillRegistryConfig", "max_ai/capabilities/skills/github/_model.py", "Defines the GitHub repository and options used to load skills."),
    ("capability-executor-local.html", "Executor", "Local executor", "Runs tool commands in the local process environment.", "max_ai/capabilities/executor/local/_executor.py · LocalExecutor", "LocalExecutorConfig", "max_ai/capabilities/executor/local/_model.py", "Defines local executor runtime options."),
    ("capability-executor-docker.html", "Executor", "Docker executor", "Runs tool commands in an isolated Docker container configured for an agent run.", "max_ai/capabilities/executor/docker/_executor.py · DockerExecutor", "DockerExecutorConfig", "max_ai/capabilities/executor/docker/_model.py", "Defines container image, resource, mount, and runtime settings."),
    ("capability-executor-modal.html", "Executor", "Modal executor", "Runs tool commands in a remote Modal environment.", "max_ai/capabilities/executor/modal/_executor.py · ModalExecutor", "ModalExecutorConfig", "max_ai/capabilities/executor/modal/_model.py", "Defines Modal app, image, and execution runtime settings."),
    ("capability-output-format.html", "Output format", "Structured output format", "Requests a final response that validates against a Pydantic model. Pass the model class to Agent as output_format.", "max_ai/agents/agent.py · Agent(output_format=...)", None, None, "There is no separate capability *Config model. The Pydantic model supplied by the application defines the output schema."),
]

PARAMETERS = {
    "model": "Provider-specific model identifier to use for requests.",
    "host": "Ollama server URL. The default points to a local Ollama instance.",
    "api_key": "Optional provider credential. Use `YOUR_API_KEY` or load the secret from an environment variable.",
    "config": "Optional ModelConfig describing model capabilities such as tool calling, vision, or token limits.",
    "think": "Ollama thinking mode: `True`, `False`, or an effort such as `low`, `medium`, or `high`.",
    "keep_alive": "How long Ollama keeps the model loaded, such as `5m`; `0` unloads it after a request.",
    "max_tokens": "Maximum number of output tokens, if you want to set a limit.",
    "api_key_env": "Name of the environment variable that contains the provider API key.",
    "kwargs": "Additional provider request options supported by the model client.",
    "base_url": "Optional API endpoint override for a compatible provider service.",
    "organization": "Optional OpenAI organization identifier.",
    "project": "Optional OpenAI project identifier.",
    "name": "Short name used to identify this component or source.",
    "description": "Human-readable explanation shown to the agent when it uses this source.",
    "base_path": "Directory used to store or read the component's local data.",
    "db_name": "Database file or database name used by this backend.",
    "tool_mode": "Controls which tools the agent can use for this capability.",
    "embedding": "Optional embedding provider used for semantic search; leave unset to disable embeddings.",
    "min_score": "Minimum relevance score for a search result to be returned.",
    "database": "MongoDB database name.",
    "collection": "MongoDB collection name.",
    "uri_env": "Environment variable containing the MongoDB connection URI.",
    "server_selection_timeout_ms": "Maximum time in milliseconds to wait for MongoDB server selection.",
    "context_days": "Number of recent days of context to include in memory operations.",
    "search_limit": "Maximum number of matching memory records to return.",
    "summary_max_tokens": "Maximum token budget for the generated conversation summary.",
    "message_cap_tokens": "Token cap applied to an individual message during compaction.",
    "max_turns": "Maximum number of recent conversation turns to retain.",
    "enabled": "Whether this completion gate is active.",
    "plan_must_close": "Whether the run must close its plan before completing.",
    "check_bash_outputs": "Whether command output is checked before the agent can finish.",
    "nudge_bash_failures": "Whether failed commands prompt the agent to address the failure.",
    "max_cost_usd": "Maximum permitted model cost in US dollars for the run.",
    "max_seconds": "Maximum duration of the run in seconds.",
    "max_model_calls": "Maximum number of model calls allowed in the run.",
    "max_tool_calls": "Maximum number of tool calls allowed in the run.",
    "quota": "Optional quota definition used to apply usage limits.",
    "quota_store": "Optional store that persists quota usage between runs.",
    "keep_periods": "Number of historical quota periods to retain.",
    "max_loop_iterations": "Maximum number of reasoning-loop iterations before stopping.",
    "guards": "Optional loop guards that check progress and completion conditions.",
    "endpoint_url": "MinIO service URL, including scheme and port when needed.",
    "bucket": "Bucket name used to store workspace files.",
    "access_key_env": "Environment variable containing the MinIO access key.",
    "secret_key_env": "Environment variable containing the MinIO secret key.",
    "region": "Optional storage region.",
    "cache_dir": "Optional local directory used to cache remote workspace files.",
    "instructions": "Instructions or behavior text supplied to this prompt layer.",
    "template": "Optional template used to render the layer content.",
    "version": "Optional version string for this layer configuration.",
    "is_edited": "Marks whether the layer content has been edited from its default.",
    "layer_class": "Optional class used to construct the configured layer.",
    "load_from": "Optional source used to load the layer configuration.",
    "extra_variables": "Additional template values available while rendering the layer.",
    "command": "Executable used to start the MCP server process.",
    "args": "Arguments passed to the MCP server process.",
    "env": "Environment variables passed to the MCP server process.",
    "server_id": "Stable identifier assigned to this MCP server.",
    "url": "URL of the remote MCP server.",
    "headers": "Optional HTTP headers sent when connecting to the server.",
    "timeout_seconds": "Maximum time to wait for an MCP operation.",
    "permission": "Rules that control which shell operations are allowed.",
    "workspace": "Workspace implementation where files are read or written.",
    "root": "Root directory or remote path used by this component.",
    "path": "Filesystem path used by this component.",
    "client": "Client or service object used to perform the component's operations.",
    "app_name": "Optional application name sent as OpenRouter request metadata.",
    "app_url": "Optional application URL sent as OpenRouter request metadata.",
    "provider": "Optional provider routing preferences for OpenRouter.",
    "fallback_models": "Optional models to try if the preferred OpenRouter model is unavailable.",
    "reasoning": "Optional reasoning effort or provider-specific reasoning settings.",
    "options": "Additional request options serialized with the component configuration.",
    "cache_path": "Optional local cache directory for remote files.",
    "image": "Container image used by the executor.",
    "app": "Optional Modal application used for remote execution.",
    "cpu": "CPU allocation requested for the execution environment.",
    "memory": "Memory allocation requested for the execution environment.",
    "timeout": "Maximum allowed runtime for a command.",
    "user_id": "Identifier for the user whose data this component stores or retrieves.",
    "session_id": "Identifier for the conversation session whose data this component stores or retrieves.",
    "level": "Minimum logging level to record.",
    "preview_chars": "Maximum number of characters included in a logged preview.",
    "capture_content": "Whether trace spans include request or response content.",
    "max_content_chars": "Maximum number of content characters recorded in a trace.",
    "tracer_provider": "Optional OpenTelemetry tracer provider; the default provider is used when omitted.",
    "max_connection_retries": "Number of times to retry a transient model connection failure.",
    "approval_mode": "Approval policy applied before the tool is run.",
    "read_only": "Marks the tool as read-only for approval and runtime policy decisions.",
    "func": "Python callable wrapped and exposed as a tool.",
    "max_retries": "Maximum number of retries after a tool execution failure.",
    "max_output_chars": "Maximum number of command output characters returned to the agent.",
    "allowed_patterns": "Command patterns that may run without an approval prompt.",
    "ask_patterns": "Command patterns that require user approval.",
    "deny_patterns": "Command patterns that are always rejected.",
    "blob_url": "Azure Blob container URL used for workspace storage.",
    "source": "Configuration identifying where the skills are loaded from.",
    "skills": "Optional set of skill names to include from the configured source.",
    "ref": "Git branch, tag, or commit to read from the repository.",
    "token_env": "Environment variable containing the GitHub access token.",
    "max_output_bytes": "Maximum command output size in bytes returned to the agent.",
    "network": "Network access policy for the execution environment.",
    "lifetime": "Maximum lifetime for the remote execution environment.",
    "output_format": "Pydantic model class that defines the required shape of the agent's final response.",
}


def inline_markup(value: str) -> str:
    """Escape table text while retaining simple inline-code spans."""
    parts = re.split(r"(`[^`]+`)", value)
    return "".join(
        f"<code>{escape(part[1:-1])}</code>" if part.startswith("`") and part.endswith("`") else escape(part)
        for part in parts
    )


def source_outline(implementation: str) -> tuple[str, list[str]]:
    """Return a class/function declaration with only its initializer signature."""
    if implementation == "max_ai/agents/agent.py · Agent(output_format=...)":
        return (
            "class ResponseSchema(BaseModel):\n    answer: str\n\n"
            "agent = Agent(\n    ...,\n    output_format=ResponseSchema,\n)",
            ["output_format"],
        )
    first = implementation.split("; ", 1)[0]
    if " · " not in first:
        return implementation, []
    source_path, symbol = first.split(" · ", 1)
    path = Path(source_path)
    if not path.exists():
        return implementation, []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    node = next(
        (item for item in ast.walk(tree) if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == symbol),
        None,
    )
    if node is None:
        return implementation, []
    params: list[str] = []
    if isinstance(node, ast.ClassDef):
        class_header = textwrap.dedent("\n".join(lines[node.lineno - 1:node.body[0].lineno - 1])).rstrip()
        init = next((item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__"), None)
        if init is None:
            return f"{class_header}\n    # Uses its inherited constructor.\n    ...", params
        signature = textwrap.dedent("\n".join(lines[init.lineno - 1:init.body[0].lineno - 1])).rstrip()
        args = [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
        params = [arg.arg for arg in args if arg.arg != "self"]
        if init.args.vararg:
            params.append("*" + init.args.vararg.arg)
        if init.args.kwarg:
            params.append("**" + init.args.kwarg.arg)
        return f"{class_header}\n{textwrap.indent(signature, '    ')}\n        ...", params
    signature = textwrap.dedent("\n".join(lines[node.lineno - 1:node.body[0].lineno - 1])).rstrip()
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    params = [arg.arg for arg in args if arg.arg not in {"self", "cls"}]
    if node.args.vararg:
        params.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        params.append("**" + node.args.kwarg.arg)
    return f"{signature}\n    ...", params


def render(row: tuple[str, ...]) -> str:
    filename, category, title, purpose, implementation, config, config_file, config_note = row
    page_id = filename.removesuffix(".html")
    config_heading = f"{config} configuration" if config else "Configuration"
    config_identity = (
        f'<p><strong>Model:</strong> <code>{escape(config)}</code><br>'
        f'<strong>Defined in:</strong> <code>{escape(config_file)}</code></p>'
        if config else '<p><strong>Dedicated config model:</strong> None.</p>'
    )
    code, parameters = source_outline(implementation)
    parameter_rows = "".join(
        f'<tr><th><code>{escape(param)}</code></th><td>{inline_markup(PARAMETERS.get(param.lstrip("*"), f"Value for {param.lstrip(chr(42))} accepted by this component."))}</td></tr>'
        for param in parameters
    )
    parameter_table = (
        f'<h3>Constructor parameters</h3><div class="table-wrap"><table class="doc-table"><thead><tr><th>Parameter</th><th>What to provide</th></tr></thead><tbody>{parameter_rows}</tbody></table></div>'
        if parameters else '<p>This implementation uses its inherited constructor; configure it through the model shown below.</p>'
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script>try {{ const saved = localStorage.getItem('max-ai-theme'); document.documentElement.dataset.theme = saved || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); }} catch (_) {{}}</script><meta name="description" content="{escape(purpose)}"><title>{escape(title)} · Max AI Docs</title><link href="/assets/styles.css" rel="stylesheet"><script defer src="/assets/app.js"></script><link href="/assets/favicon.svg" rel="icon" type="image/svg+xml"></head>
<body data-page="{escape(page_path(filename))}"><a class="skip-link" href="#main-content">Skip to content</a>
<header class="topbar"><a class="brand" href="/overview/index.html" aria-label="Max AI home"><span>max<span class="brand-accent">ai</span></span><span class="brand-divider"></span><span class="header-label">Documentation</span></a><div class="topbar-actions"><button class="search-trigger" type="button" aria-controls="search-dialog" aria-haspopup="dialog"><span>Search documentation</span><kbd>⌘ K</kbd></button><button class="icon-button" id="theme-toggle" type="button" aria-label="Switch color theme">◐</button><button class="menu-button" id="menu-toggle" type="button" aria-controls="sidebar" aria-expanded="false" aria-label="Open navigation">☰</button></div></header>
<nav class="tabs-bar" aria-label="Primary"><a class="selected" href="/overview/index.html">Documentation</a><a href="/overview/index.html">Installation</a><a href="/agents/index.html#agent">API guide</a><span class="release-label"><span class="status-dot"></span>v0.1.0 · in development</span></nav>
<aside class="sidebar" id="sidebar"><div class="sidebar-title">Max AI documentation</div><nav aria-label="Documentation"></nav></aside><button class="nav-backdrop" type="button" tabindex="-1" aria-label="Close navigation"></button>
<main class="main-content" id="main-content" tabindex="-1"><nav class="breadcrumb" aria-label="Breadcrumb"><a href="/overview/index.html">Docs</a><span>›</span><span>Capabilities</span><span>›</span><span>{escape(category)}</span><span>›</span><span>{escape(title)}</span></nav>
<details class="mobile-toc"><summary>On this page</summary><nav aria-label="Page sections"><a class="toc-link" href="#{page_id}-overview">Overview</a><a class="toc-link" href="#{page_id}-implementation">Implementation</a><a class="toc-link" href="#{page_id}-configuration">Configuration</a></nav></details>
<section class="hero content-section page-hero" id="{page_id}-top"><p class="eyebrow">Capabilities / {escape(category)}</p><h1>{escape(title)}</h1><p class="hero-copy">{escape(purpose)}</p></section>
<section class="section-block content-section" id="{page_id}-overview"><h2>What it does<a class="heading-anchor" href="#{page_id}-overview">#</a></h2><p>{escape(purpose)}</p></section>
<section class="section-block content-section" id="{page_id}-implementation"><h2>Implementation<a class="heading-anchor" href="#{page_id}-implementation">#</a></h2><p>Class declaration and constructor signature:</p><div class="code-card"><div class="code-head"><span>Python · {escape(Path(implementation.split(' · ', 1)[0].split('; ', 1)[0]).name)}</span></div><pre tabindex="0"><code>{escape(code)}</code></pre></div>{parameter_table}</section>
<section class="section-block content-section" id="{page_id}-configuration"><h2>{escape(config_heading)}<a class="heading-anchor" href="#{page_id}-configuration">#</a></h2>{config_identity}<p>{escape(config_note)}</p></section>
</main>
<dialog aria-labelledby="search-title" class="search-dialog" id="search-dialog"><h2 class="sr-only" id="search-title">Search documentation</h2><div class="search-input-row"><label class="sr-only" for="docs-search">Search documentation</label><input autocomplete="off" id="docs-search" placeholder="Search documentation…" spellcheck="false" type="search"><button aria-label="Close search" id="search-close" type="button">Esc</button></div><div class="search-results" id="search-results"></div><p class="sr-only" id="search-status" role="status"></p><div class="search-footer"><span><kbd>↑</kbd> <kbd>↓</kbd> navigate　<kbd>↵</kbd> open</span><span>Search Max AI docs</span></div></dialog>
</body></html>'''


def main() -> None:
    for row in PAGES:
        target = HERE / page_path(row[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(row), encoding="utf-8")
    print(f"Generated {len(PAGES)} capability pages.")


if __name__ == "__main__":
    main()
