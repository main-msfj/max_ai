"""Generate compact reference pages for runtime types and events."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHELL = (ROOT / "cli" / "index.html").read_text(encoding="utf-8")

PAGES = {
    "types/agent-response.html": {
        "title": "AgentResponse",
        "description": "The result returned by Agent.run(), including its context, completion state, and usage.",
        "id": "agent-response",
        "headline": "A structured result returned when an agent run ends.",
        "copy": "AgentResponse is the value returned when an agent run ends. It carries the run context, aggregate usage, finish reason, and optional completion decision. Use it to read the final message, check whether the run paused for approval or input, and resume from its context.",
        "code": '''class AgentResponse(BaseModel):
    context: RunContext | None
    source: str
    usage: Usage
    finish_reason: FinishReason
    completion: CompletionDecision | None = None
    stop_message: str | None = None
    created_at: datetime''',
        "fields": [
            ("context", "Conversation and runtime state. Pass it to Agent.resume() to continue a paused run."),
            ("source", "Name of the agent that produced this response."),
            ("usage", "Aggregated timing, model calls, tool calls, and token counts."),
            ("finish_reason", "Why the run stopped, such as stop, approval_needed, or cancelled."),
            ("completion", "The final completion gate decision, when one was produced."),
            ("final_message / final_text", "Convenience properties for accessing the user-facing answer."),
        ],
    },
    "types/completion.html": {
        "title": "Completion",
        "description": "CompletionDecision describes whether an agent response is complete, waiting, or needs another turn.",
        "id": "completion",
        "headline": "The outcome of checking whether the agent can finish its turn.",
        "copy": "Completion types describe the result of checking a proposed final response. CompletionDecision is the aggregate outcome returned by completion gates and attached to AgentResponse. A decision can be completed, incomplete, waiting, or cancelled, and can include reasons that explain the outcome.",
        "code": '''class Usage(BaseModel):
    duration_ms: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0

class ChatCompletionResult(BaseModel):
    message: AssistantMessage
    usage: Usage
    model: str
    finish_reason: str

class ChatCompletionChunk(BaseModel):
    content: str
    is_complete: bool
    thinking: str | None = None

class CompletionDecision(BaseModel):
    status: Literal["completed", "incomplete", "waiting", "cancelled"]
    reasons: tuple[str, ...] = ()''',
        "fields": [
            ("Usage", "Aggregated time, model and tool calls, and token counts."),
            ("ChatCompletionResult", "Normalized, non-streaming model output with its usage."),
            ("ChatCompletionChunk", "One partial model output item when streaming."),
            ("CompletionDecision.status", "The outcome of the agent's final response check."),
            ("CompletionDecision.reasons", "Messages explaining why completion was accepted or deferred."),
            ("CompletionCheck", "A single check result used when a gate combines multiple checks."),
            ("CompletionBase", "The base contract implemented by completion gate components."),
        ],
    },
    "types/tool.html": {
        "title": "Tool",
        "description": "Tool types describe callable capabilities and their parameters in Max AI.",
        "id": "tool",
        "headline": "The contract and metadata for actions an agent can request.",
        "copy": "A tool gives the agent a defined action it can request during a run. CoreTool is the component contract for tool implementations. CoreToolDefinition describes the name, purpose, and parameter schema exposed to the model; CoreToolParameters carries validation state for execution.",
        "code": '''class CoreToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = {}

class CoreToolParameters(BaseModel):
    is_tool_valid: bool = False
    msg_error: str | None = None''',
        "fields": [
            ("name", "Tool name exposed to the model."),
            ("description", "Short explanation of what the tool does."),
            ("parameters", "JSON schema describing the arguments the tool accepts."),
            ("is_tool_valid", "Whether the submitted parameters passed validation."),
            ("msg_error", "Validation message when the parameters are invalid."),
        ],
    },
    "types/tool-call.html": {
        "title": "Tool call",
        "description": "ToolCall, ToolCallRecord, and ToolResult represent a requested action and its execution lifecycle.",
        "id": "tool-call",
        "headline": "The request, execution state, and result for one tool action.",
        "copy": "A tool call starts as a model request containing a tool name and parameters. ToolCallRecord tracks that request through approval and execution. ToolResult stores the outcome and is attached to the record, so applications can inspect both what was requested and what happened.",
        "code": '''class ToolCall(BaseModel):
    id: str
    tool_name: str
    parameters: dict[str, Any]

class ToolResult(BaseModel):
    success: bool
    error: str | None = None
    result: Any | None = None
    tool_call_id: str''',
        "fields": [
            ("ToolCall", "The assistant's requested action: identifier, tool name, and arguments."),
            ("ToolCallRecord", "The persisted lifecycle record, including status, approval, timing, and result."),
            ("ToolResult", "The execution output or failure, linked to its originating call."),
            ("status", "Tracks pending approval, approved, executing, or consumed states."),
        ],
    },
    "types/run-context.html": {
        "title": "RunContext",
        "description": "RunContext stores the conversation and runtime state used to continue an agent session.",
        "id": "run-context",
        "headline": "The state an agent run reads, updates, and can resume from.",
        "copy": "RunContext holds the state for a user's conversation and the current run. It combines the current messages with saved history, tool-call state, runtime state, plans, compaction state, and completion-gate evidence. AgentResponse exposes this context so a paused run can continue with the same state.",
        "code": '''class RunContext(BaseModel):
    user_id: str
    session_id: str | None
    run_id: str
    messages: list[Message]
    message_history: ChatHistory
    tool_state: ToolState
    runtime_state: RuntimeState
    plan: AgentPlan | None
    compaction: CompactionState
    completion_state: dict[str, dict[str, JsonValue]]''',
        "fields": [
            ("user_id / session_id / run_id", "Identify the user, resumable conversation, and individual run."),
            ("messages", "Messages added during the current task or turn."),
            ("message_history", "Earlier conversation messages loaded for the session."),
            ("tool_state", "Tool calls, approvals, pending questions, execution status, and results."),
            ("runtime_state", "Runtime metadata, environment data, and shared state."),
            ("plan", "An agent plan, when the agent is using one."),
            ("compaction", "State used to manage and summarize the active conversation context."),
            ("completion_state", "Evidence recorded by completion gates; preserved across pause and resume."),
        ],
    },
}

EVENT_GROUPS = [
    ("Tasks", "Task boundaries and whether a proposed final answer passed the completion gate.", [
        ("TaskStartEvent", "task_start", "Processing of a task begins.", "Includes the task text."),
        ("TaskCompleteEvent", "task_complete", "The completion gate accepted the final response.", "Includes the aggregate completion decision. It is not emitted for a pause, error, or exhausted limit."),
        ("CompletionRejectedEvent", "completion_rejected", "The completion gate rejected a proposed final response, so the model can try again.", "Includes the decision and its reasons."),
    ]),
    ("Model", "Model request, response, and streaming output activity.", [
        ("ModelCallEvent", "model_call", "A request is being sent to a model.", "Includes the model name, input messages, and rendered prompt token count."),
        ("ModelResponseEvent", "model_response", "A model response has arrived.", "Includes response text, whether it contains tool calls, and usage when available."),
        ("ModelStreamChunkEvent", "model_stream_chunk", "A partial model output is available during streaming.", "Includes the text chunk, optional thinking, and whether it is the final chunk."),
    ]),
    ("Reasoning", "Reasoning loop progress, planning, and interactions that pause for the user.", [
        ("ReasoningIterationEvent", "reasoning_iteration", "A reasoning loop iteration begins.", "Includes the current iteration and maximum allowed iterations."),
        ("ReasoningCompleteEvent", "reasoning_complete", "The reasoning loop has stopped.", "Includes its finish reason and number of iterations."),
        ("LastMessageResponseEvent", "last_message_response", "The model proposed a final text response without requesting tools; the completion gate is about to check it.", "Includes the proposed response."),
        ("PlanningEvent", "planning", "A plan is starting, changing, finishing, failing, or being skipped.", "Includes the phase and plan when available."),
        ("UserInputRequestEvent", "user_input_request", "The agent is asking the user a question and pausing for an answer.", "Includes the question, optional choices, and tool call ID used to store the answer."),
    ]),
    ("Agent", "Agent selection, execution boundaries, and changes to the active context window.", [
        ("AgentSelectionEvent", "agent_selection", "An agent was selected to handle work.", "Includes the selected agent and optional selection reason."),
        ("AgentExecutionStartEvent", "agent_execution_start", "An agent begins execution.", "Includes the agent name and context size."),
        ("AgentExecutionCompleteEvent", "agent_execution_complete", "An agent finishes execution.", "Includes the agent name, success status, and produced message count."),
        ("CompactionEvent", "compaction", "Context compaction begins or ends.", "Includes phase, strategy, token counts, whether the context changed, and at the end the removed messages and summary."),
    ]),
    ("Tools and commands", "Tool requests, approval, validation, progress, command execution, and workspace file operations.", [
        ("ToolCallEvent", "tool_call", "A tool call is about to be handled.", "Includes tool name, parameters, and tool call ID. This describes the request, not its result."),
        ("ToolApprovalEvent", "tool_approval", "A tool needs a user's approval before it can run.", "Includes the tool, parameters, call ID, and optional approval reason."),
        ("ToolAutoApprovalEvent", "tool_auto_approval", "Policy approved a tool without asking the user.", "Includes tool name and call ID."),
        ("ToolValidationEvent", "tool_validation", "Tool parameters have been checked.", "Includes whether they are valid and any validation errors."),
        ("ToolProgressEvent", "tool_progress", "A long-running tool reports progress.", "Includes a human-readable update, tool name, and call ID."),
        ("ToolCallResponseEvent", "tool_call_response", "Tool execution finished, successfully or with a failure.", "Carries the ToolResult and originating call ID. This is the tool's runtime outcome."),
        ("BashStartedEvent", "bash_started", "A shell command is about to execute.", "Includes the command, declared action, description, and call ID."),
        ("BashFinishedEvent", "bash_finished", "A shell command finished.", "Includes exit code, duration, timeout status, and whether output was truncated."),
        ("BashFailedEvent", "bash_failed", "A shell command failed to start or run.", "Includes the call ID and failure message."),
        ("BashCancelledEvent", "bash_cancelled", "A shell command was cancelled.", "Includes the call ID and cancellation reason."),
        ("FileReadEvent", "file_read", "A workspace file was read successfully.", "Includes its path, workspace root, content hash, and call ID."),
        ("DirectoryListedEvent", "directory_listed", "A workspace directory was listed successfully.", "Includes its path, workspace root, entry count, and call ID."),
        ("FileWrittenEvent", "file_written", "A workspace file was written or edited successfully.", "Includes operation, path, workspace root, content hash, and call ID."),
        ("FilesSearchedEvent", "files_searched", "A workspace file or text search finished.", "Includes search operation, path, match count, truncation status, and call ID."),
        ("DirectoryCreatedEvent", "directory_created", "A workspace directory was created.", "Includes its path, workspace root, and call ID."),
        ("FileDeletedEvent", "file_deleted", "A workspace file was deleted.", "Includes its path, workspace root, content hash, and call ID."),
        ("FileInfoEvent", "file_info", "Information about a workspace file or directory was retrieved.", "Includes the path, workspace root, item type, and call ID."),
    ]),
    ("Errors", "Recoverable problems and failures that stop execution.", [
        ("ErrorEvent", "error", "A recoverable runtime error occurred.", "Includes error text, category, and recoverability."),
        ("FatalErrorEvent", "fatal_error", "An unrecoverable error is terminating execution.", "Includes error text and category; recoverability is false."),
    ]),
    ("Memory", "Memory access and updates performed during a run.", [
        ("MemoryUpdateEvent", "memory_update", "Memory state changed.", "Includes the operation that changed it."),
        ("MemoryRetrievalEvent", "memory_retrieval", "The agent retrieved memory entries.", "Includes the search query and number of results."),
    ]),
]

EVENTS = {
    "title": "Events",
    "description": "Runtime event types emitted while an agent processes a run.",
    "id": "events",
    "headline": "Typed records that describe what happens during an agent run.",
    "copy": "This reference describes agent runtime events, grouped by what they communicate. Each event carries details a host can use to update a UI, log activity, or react to a pause. Orchestration events are left out of this guide.",
}


def render(data: dict[str, object]) -> str:
    if data.get("id") == "events":
        return render_events(data)
    page_id = str(data["id"])
    toc = ["Overview", "Type outline", "Fields and related types"]
    toc_html = "".join(f'<a class="toc-link" href="#{page_id}-{i}">{label}</a>' for i, label in enumerate(toc))
    rows = "".join(f"<tr><td><code>{name}</code></td><td>{description}</td></tr>" for name, description in data["fields"])
    main = f'''<main class="main-content" id="main-content" tabindex="-1">
<nav aria-label="Breadcrumb" class="breadcrumb"><a href="/overview/index.html">Docs</a><span>›</span><span>{data['title']}</span></nav>
<details class="mobile-toc"><summary>On this page</summary><nav aria-label="Page sections">{toc_html}</nav></details>
<section class="hero content-section page-hero" id="{page_id}-top"><h1>{data['title']}</h1><p class="hero-copy">{data['headline']}</p></section>
<section class="section-block content-section" id="{page_id}-0"><h2>Overview</h2><p>{data['copy']}</p></section>
<section class="section-block content-section" id="{page_id}-1"><h2>Type outline</h2><div class="code-card"><div class="code-head"><span>Python types</span></div><pre tabindex="0"><code>{data['code']}</code></pre></div></section>
<section class="section-block content-section" id="{page_id}-2"><h2>Fields and related types</h2><div class="table-wrap"><table class="doc-table"><thead><tr><th>Type or field</th><th>Purpose</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<nav aria-label="Previous and next page" class="page-pagination"><a href="/agents/index.html"><small>← Previous</small><strong>Agent</strong></a><a class="next" href="/events/index.html"><small>Next →</small><strong>Events</strong></a></nav>
<footer class="footer"><span>Max AI · Python agent framework</span></footer></main>'''
    html = re.sub(r'<main class="main-content".*?</main>', main, SHELL, count=1, flags=re.DOTALL)
    html = re.sub(r'<aside aria-label="On this page" class="page-toc">.*?</aside>', f'<aside aria-label="On this page" class="page-toc"><p>On this page</p><nav>{toc_html}</nav><div class="toc-extra"><a href="#{page_id}-top">Back to top ↑</a></div></aside>', html, count=1, flags=re.DOTALL)
    html = html.replace('content="Max AI CLI reference: run and interact with an agent from the terminal."', f'content="{data["description"]}"')
    html = html.replace("<title>CLI · Max AI Docs</title>", f"<title>{data['title']} · Max AI Docs</title>")
    html = html.replace('data-page="cli.html"', f'data-page="{page_id}.html"')
    return html


def render_events(data: dict[str, object]) -> str:
    page_id = "events"
    code_example = '''from max_ai.types.agent_response import AgentResponse

async def stream_to_ui(agent, ui):
    async for item in agent.run_stream_events(
        "Summarize the selected document",
        stream_tokens=True,
    ):
        if isinstance(item, AgentResponse):
            ui.show_answer(item.final_text)
        else:
            ui.add_activity(item.event_type, item)'''
    event_ui = '''<div class="event-ui-preview" aria-label="Illustrative agent activity feed">
<div class="event-ui-head"><strong>Agent run</strong><span>Live activity</span></div>
<div class="event-ui-list">
<div class="event-ui-row"><span class="event-ui-dot"></span><div><strong>Reasoning step started</strong><small>reasoning_iteration</small></div><time>10:24:10</time></div>
<div class="event-ui-row"><span class="event-ui-dot"></span><div><strong>Model request</strong><small>model_call</small></div><time>10:24:10</time></div>
<div class="event-ui-row"><span class="event-ui-dot"></span><div><strong>Tool requested</strong><small>tool_call · registered tool</small></div><time>10:24:11</time></div>
<div class="event-ui-row"><span class="event-ui-dot"></span><div><strong>Tool finished</strong><small>tool_call_response · result received</small></div><time>10:24:12</time></div>
<div class="event-ui-row"><span class="event-ui-dot"></span><div><strong>Assistant response streaming</strong><small>model_stream_chunk</small></div><time>10:24:13</time></div>
<div class="event-ui-row"><span class="event-ui-dot event-ui-dot-done"></span><div><strong>Answer accepted</strong><small>task_complete</small></div><time>10:24:13</time></div>
</div><div class="event-ui-answer"><span>AgentResponse · final_text</span><p>The selected document summarizes the project status and next steps.</p></div></div>'''
    toc_items = [("How to read the stream", "stream"), ("Example UI", "example-ui"), ("Common event fields", "common-fields")]
    toc_items.extend((name, "family-" + name.lower().replace(" ", "-")) for name, _, _ in EVENT_GROUPS)
    toc_html = "".join(f'<a class="toc-link" href="#{anchor}">{escape(name)}</a>' for name, anchor in toc_items)
    common_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{description}</td></tr>"
        for name, description in [
            ("source", "The component that emitted the event."),
            ("event_type", "Stable event name, such as tool_call or model_response."),
            ("event_id", "Unique identifier for this event."),
            ("timestamp", "When the event was created."),
        ]
    )
    family_sections = []
    count = 0
    for family, summary, events in EVENT_GROUPS:
        count += len(events)
        rows = "".join(
            f'<tr><td><code>{escape(class_name)}</code><br><span class="muted">{escape(event_type)}</span></td>'
            f'<td>{escape(meaning)}</td><td>{escape(details)}</td></tr>'
            for class_name, event_type, meaning, details in events
        )
        anchor = "family-" + family.lower().replace(" ", "-")
        family_sections.append(
            f'<section class="section-block content-section" id="{anchor}"><h2>{escape(family)}</h2>'
            f'<p>{escape(summary)}</p><div class="table-wrap"><table class="doc-table"><thead>'
            f'<tr><th>Event</th><th>What it expresses</th><th>Useful data</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></section>'
        )
    main = f'''<main class="main-content" id="main-content" tabindex="-1">
<nav aria-label="Breadcrumb" class="breadcrumb"><a href="/overview/index.html">Docs</a><span>›</span><span>Events</span></nav>
<details class="mobile-toc"><summary>On this page</summary><nav aria-label="Page sections">{toc_html}</nav></details>
<section class="hero content-section page-hero" id="events-top"><h1>Events</h1><p class="hero-copy">A reference to {count} agent runtime events and what each one communicates. Orchestration events are not included here.</p></section>
<section class="section-block content-section" id="stream"><h2>How to read the stream</h2><p><code>Agent.run_stream_events()</code> yields typed <code>CoreEvent</code> objects as work happens, then yields one <code>AgentResponse</code> when the run ends. <code>AgentResponse</code> is the final run result, not an event.</p><p>A model's final-looking text is not necessarily an accepted answer: <code>LastMessageResponseEvent</code> describes a proposed answer, <code>CompletionRejectedEvent</code> means the completion gate wants another attempt, and <code>TaskCompleteEvent</code> means the gate accepted the response.</p><p>Tool events use the same <code>tool_call_id</code> to connect a request with its approval, progress, and result. <code>ToolCallEvent</code> describes the requested action; <code>ToolCallResponseEvent</code> carries its <code>ToolResult</code>. The result may also appear in the conversation as a <code>ToolMessage</code>.</p><div class="code-card"><div class="code-head"><span>Send runtime activity to your UI</span></div><pre tabindex="0"><code>{escape(code_example)}</code></pre></div><p>The example leaves <code>agent</code> and <code>ui</code> to the application: the agent is already configured, and the UI decides how to display each event.</p></section>
<section class="section-block content-section" id="example-ui"><h2>Example UI</h2><p>A host can turn the stream into a small activity feed while the final answer appears when the <code>AgentResponse</code> arrives.</p>{event_ui}<p>This is an illustrative presentation. Your application can show fewer details, expose tool approval controls, or display model chunks as they arrive.</p></section>
<section class="section-block content-section" id="common-fields"><h2>Common event fields</h2><p>Every event inherits these fields from <code>CoreEvent</code>:</p><div class="table-wrap"><table class="doc-table"><thead><tr><th>Field</th><th>Meaning</th></tr></thead><tbody>{common_rows}</tbody></table></div></section>
{''.join(family_sections)}
<nav aria-label="Previous and next page" class="page-pagination"><a href="/types/tool-call.html#tool-call-top"><small>← Previous</small><strong>Tool call</strong></a><a class="next" href="/cli/index.html#cli-top"><small>Next →</small><strong>CLI</strong></a></nav>
<footer class="footer"><span>Max AI · Python agent framework</span></footer></main>'''
    html = re.sub(r'<main class="main-content".*?</main>', main, SHELL, count=1, flags=re.DOTALL)
    html = re.sub(r'<aside aria-label="On this page" class="page-toc">.*?</aside>', f'<aside aria-label="On this page" class="page-toc"><p>On this page</p><nav>{toc_html}</nav><div class="toc-extra"><a href="#events-top">Back to top ↑</a></div></aside>', html, count=1, flags=re.DOTALL)
    html = html.replace('content="Max AI CLI reference: run and interact with an agent from the terminal."', f'content="{escape(str(data["description"]))}"')
    html = html.replace("<title>CLI · Max AI Docs</title>", "<title>Events · Max AI Docs</title>")
    html = html.replace('data-page="cli.html"', 'data-page="events/index.html"')
    return html


def main() -> None:
    for relative, data in PAGES.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        html = render(data).replace(f'data-page="{data["id"]}.html"', f'data-page="{relative}"')
        path.write_text(html, encoding="utf-8")
    events_path = ROOT / "events/index.html"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(render(EVENTS).replace('data-page="events.html"', 'data-page="events/index.html"'), encoding="utf-8")
    print(f"Generated {len(PAGES) + 1} type and event pages.")


if __name__ == "__main__":
    main()
