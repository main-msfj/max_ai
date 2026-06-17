"""Interactive chat REPL for a max_ai agent.

``run_repl(agent)`` opens a conversation in the terminal:

  * reads a line from the user,
  * streams the agent's events through ``CliRenderer``,
  * answers the agent's own questions inline (the native
    ``structure_human_in_loop`` tool) without ending the turn,
  * threads the run's ``RunContext`` into the next turn so the agent
    remembers the conversation.

Concurrency note — why this is not a plain ``async for`` over the stream:
the reasoning loop blocks on ``await future`` *inside* the human-input tool
*before* it emits ``UserInputRequestEvent``. A single-process consumer that
just iterates the stream would deadlock: it can't read the question (the
event hasn't been yielded) and it can't answer (it's awaiting the next
event). So we drive the event stream in a background task and, in the
foreground, watch ``agent.pending_user_input``; when a question appears we
prompt the user and call ``agent.provide_user_input`` to unblock the loop.

All rendering lives in ``CliRenderer``; this module owns only the loop and
the human-input handshake.
"""

from __future__ import annotations

import asyncio
import typing as t

from rich.console import Console
from rich.text import Text
from rich.panel import Panel

from ..base.agent import Agent
from ..core.messages import AssistantMessage, ToolMessage
from ..types.run_context import RunContext
from ..types.agent_response import AgentResponse
from .renderer import CliRenderer


_EXIT_COMMANDS = {"/exit", "/quit", "/q"}

# How often the foreground checks for a pending agent question while the
# event stream runs in the background.
_POLL_INTERVAL = 0.05


async def run_repl(
    agent: Agent,
    *,
    console: Console | None = None,
    show_thinking: bool = True,
) -> None:
    """Run the interactive chat loop until the user exits.

    Args:
        agent: A constructed agent. Its reasoning loop drives each turn; the
            human-input tool is registered natively, so the agent can ask the
            user questions without any extra wiring.
        console: Optional Rich console (mainly for tests).
        show_thinking: Render the model's reasoning tokens (dimmed).
    """
    renderer = CliRenderer(console=console, show_thinking=show_thinking)
    renderer.banner(agent.name)

    # Carried across turns so the agent sees the full conversation. The first
    # turn starts fresh (None); every turn after reuses the terminal context.
    ctx: RunContext | None = None

    while True:
        try:
            user_text = (await _ainput(renderer, Text("\nyou  ", style="bold green"))).strip()
        except (EOFError, KeyboardInterrupt):
            renderer.console.print("\n[dim]bye[/dim]")
            return

        if not user_text:
            continue
        if user_text.lower() in _EXIT_COMMANDS:
            renderer.console.print("[dim]bye[/dim]")
            return

        # Defensive: if the previous turn ended with a tool call that never
        # got a response (e.g. it failed/cancelled mid-flight), the transcript
        # has an orphaned assistant tool_call. Sending that to the provider on
        # the next turn is a hard 400 ("tool_calls must be followed by tool
        # messages"). Drop those orphans but keep the rest of the conversation.
        if ctx is not None:
            _sanitize_orphan_tool_calls(ctx)

        try:
            ctx = await _run_one_turn(agent, user_text, ctx, renderer)
        except asyncio.CancelledError:
            renderer.console.print("\n[yellow]interrupted[/yellow]")
            renderer.end_turn()


async def _run_one_turn(
    agent: Agent,
    task: str,
    ctx: RunContext | None,
    renderer: CliRenderer,
) -> RunContext | None:
    """Drive a single user turn to completion.

    Returns the terminal ``RunContext`` to thread into the next turn.

    A turn can pause twice over: for a human-input question (handled inline
    while the stream runs, see ``_drive_stream``) and for tool approval (the
    stream ends with ``finish_reason='approval_needed'``; we prompt y/n, apply
    the decisions, and resume). The approval loop repeats until the turn
    finishes for real or the user rejects everything.
    """
    renderer.begin_turn()

    # First segment: the fresh task.
    response = await _drive_stream(
        agent,
        agent.run_stream_events(task=task, run_context=ctx, stream_tokens=True),
        renderer,
    )

    # Approval loop: while the turn paused for tool approval, resolve it and
    # resume the same context until the turn completes.
    while response is not None and response.needs_approval:
        resolved = await _resolve_approvals(response, renderer)
        if not resolved:
            break  # nothing got approved and we can't make progress
        response = await _drive_stream(
            agent,
            agent.resume_stream_events(
                run_context=response.context, stream_tokens=True
            ),
            renderer,
        )

    renderer.end_turn()
    return response.context if response is not None else ctx


async def _drive_stream(
    agent: Agent,
    stream: t.AsyncGenerator[t.Any, None],
    renderer: CliRenderer,
) -> AgentResponse | None:
    """Consume one event-stream segment, answering human-input questions inline.

    The stream is drained in a background task while this coroutine polls
    ``agent.pending_user_input`` (the loop blocks on a future *before* emitting
    the input event, so we can't learn of a question from the stream itself —
    see the module docstring). Returns the segment's terminal ``AgentResponse``
    (``None`` if the stream raised before producing one).
    """
    response: AgentResponse | None = None

    async def _consume() -> None:
        nonlocal response
        async for item in stream:
            if isinstance(item, AgentResponse):
                response = item
                continue
            renderer.handle(item)

    consumer = asyncio.create_task(_consume())

    # Track questions we've already answered so we prompt once per question.
    answered_question: str | None = None
    while not consumer.done():
        question = agent.pending_user_input
        if question is not None and question != answered_question:
            options = agent.pending_user_input_options
            answer = await _ask_user(renderer, question, options)
            agent.provide_user_input(answer)
            answered_question = question
        await asyncio.sleep(_POLL_INTERVAL)

    # Drain the finished task. The loop already emits an ErrorEvent for any
    # failure (rendered above), so a raised exception here would be a duplicate
    # crashing the whole REPL with a raw traceback. Swallow it and keep the
    # conversation going — the user saw the readable error line already.
    exc = consumer.exception()
    if exc is not None and not isinstance(exc, asyncio.CancelledError):
        renderer.show_error(str(exc))

    return response


async def _resolve_approvals(
    response: AgentResponse,
    renderer: CliRenderer,
) -> bool:
    """Prompt the user to approve/reject each pending tool call.

    Applies every decision to ``response.context.tool_state`` so the resumed
    run sees them. Returns ``True`` if at least one tool was approved (i.e.
    there is work to resume), ``False`` if the user rejected everything.
    """
    ctx = response.context
    if ctx is None:
        return False

    any_approved = False
    for record in response.pending_approvals:
        approved = await _ask_approval(renderer, record.tool_name, record.parameters)
        ctx.tool_state.apply_approval(record.id, approved=approved)
        any_approved = any_approved or approved
    return any_approved


def _sanitize_orphan_tool_calls(ctx: RunContext) -> None:
    """Drop assistant messages whose tool calls were never answered.

    A turn can end with an ``AssistantMessage`` carrying ``tool_calls`` that
    have no matching ``ToolMessage`` (the tool failed or the run was cut off
    before the result was recorded). Replaying that transcript to a provider
    is an immediate 400. We remove only the orphaned assistant messages (and
    keep everything else), so the conversation continues with clean context.

    Mutates ``ctx.messages`` in place.
    """
    messages = ctx.messages
    answered: set[str] = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    cleaned = [
        m
        for m in messages
        if not (
            isinstance(m, AssistantMessage)
            and m.tool_calls
            and any(tc.id not in answered for tc in m.tool_calls)
        )
    ]
    if len(cleaned) != len(messages):
        ctx.messages[:] = cleaned


async def _ask_user(
    renderer: CliRenderer,
    question: str,
    options: list[str] | None,
) -> str:
    """Show the agent's question and read the user's answer."""
    renderer.show_question(question, options)
    answer = await _ainput(renderer, Text("answer  ", style="bold green"))
    # If options were offered and the user typed a number, map it to the choice.
    if options:
        stripped = answer.strip()
        if stripped.isdigit():
            idx = int(stripped) - 1
            if 0 <= idx < len(options):
                return options[idx]
    return answer.strip()


async def _ask_approval(
    renderer: CliRenderer,
    tool_name: str,
    parameters: dict[str, t.Any],
) -> bool:
    """Show a tool the agent wants to run and read a y/n decision.

    Anything starting with 'y' (default on empty) approves; otherwise rejects.
    """
    renderer.show_approval_request(tool_name, parameters)
    answer = (await _ainput(renderer, Text("approve? [Y/n]  ", style="bold yellow"))).strip().lower()
    return answer in ("", "y", "yes")


async def _ainput(renderer: CliRenderer, prompt: t.Any) -> str:
    """Read one line of input without blocking the event loop.

    ``console.input`` is blocking; running it in a thread keeps the
    background event stream alive while the user types.
    """
    return await asyncio.to_thread(renderer.console.input, prompt)
