"""Interactive chat REPL for a max_ai agent.

``run_repl(agent)`` opens a conversation in the terminal:

  * reads a line from the user,
  * streams the agent's events through ``CliRenderer``,
  * resolves the agent's pauses between segments — tool approvals *and*
    the agent's own questions (the native ``ask_user``
    tool) are both pure record state on ``ctx.tool_state``, so the flow
    is identical: the segment ends, we prompt the user, apply the
    decision/answer, and ``resume`` the same context,
  * threads the run's ``RunContext`` into the next turn so the agent
    remembers the conversation.

There is no polling and no background task: the executor emits
``UserInputRequestEvent`` *instead of* executing the ask-the-user tool,
so the stream simply ends with ``finish_reason='input_needed'`` and the
question is waiting on the terminal ``AgentResponse``.
"""

from __future__ import annotations

import asyncio
import typing as t

from rich.console import Console
from rich.text import Text

from ..base.agent import Agent
from ..core.messages import AssistantMessage, ToolMessage
from ..types.run_context import RunContext
from ..types.agent_response import AgentResponse
from .renderer import CliRenderer


_EXIT_COMMANDS = {"/exit", "/quit", "/q"}


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

    A turn can pause for tool approval and for the agent's own questions.
    Both end the stream segment; we resolve whatever is pending on the
    terminal response and resume the same context until the turn finishes
    for real (or the user rejects everything).
    """
    renderer.begin_turn()

    # First segment: the fresh task.
    response = await _drive_stream(
        agent,
        agent.run_stream_events(task=task, run_context=ctx, stream_tokens=True),
        renderer,
    )

    # Pause loop: while the turn paused for approvals or questions, resolve
    # them and resume the same context until the turn completes.
    while response is not None and (response.needs_approval or response.needs_input):
        resolved = await _resolve_pauses(response, renderer)
        if not resolved:
            break  # nothing got approved/answered and we can't make progress
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
    """Consume one event-stream segment.

    Returns the segment's terminal ``AgentResponse`` (``None`` if the
    stream raised before producing one). The loop already emits an
    ErrorEvent for any failure (rendered inline), so a raised exception
    is swallowed after showing a readable error line — the REPL keeps
    the conversation going.
    """
    response: AgentResponse | None = None
    try:
        async for item in stream:
            if isinstance(item, AgentResponse):
                response = item
                continue
            renderer.handle(item)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        renderer.show_error(str(exc))

    return response


async def _resolve_pauses(
    response: AgentResponse,
    renderer: CliRenderer,
) -> bool:
    """Prompt the user for every pending approval and question.

    Applies every decision/answer to ``response.context.tool_state`` so the
    resumed run sees them. Returns ``True`` if there is work to resume (at
    least one approval granted or question answered), ``False`` otherwise.
    """
    ctx = response.context
    if ctx is None:
        return False

    made_progress = False

    for record in response.pending_approvals:
        approved = await _ask_approval(renderer, record.tool_name, record.parameters)
        ctx.tool_state.apply_approval(record.id, approved=approved)
        made_progress = made_progress or approved

    for record in response.pending_questions:
        answer = await _ask_user(
            renderer, record.input_question or "", record.input_options
        )
        ctx.tool_state.apply_user_answer(record.id, answer)
        made_progress = True

    return made_progress


def _sanitize_orphan_tool_calls(ctx: RunContext) -> None:
    """Drop assistant messages whose tool calls were never answered.

    A turn can end with an ``AssistantMessage`` carrying ``tool_calls`` that
    have no matching ``ToolMessage`` (the tool failed or the run was cut off
    before the result was recorded). Replaying that transcript to a provider
    is an immediate 400. We remove only the orphaned assistant messages (and
    keep everything else), so the conversation continues with clean context.

    Records still pending on ``ctx.tool_state`` (approvals, questions) are
    resolved between segments, never across user turns — by the time a new
    user turn starts they are either consumed or their assistant message is
    an orphan handled here.

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
    rest of the terminal responsive while the user types.
    """
    return await asyncio.to_thread(renderer.console.input, prompt)
