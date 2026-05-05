"""
Interactive terminal chat for MaxAI agents.

Step 1: bare REPL plumbing.
Step 2: streaming events — live LLM chunks + tool call indicators.
Step 4: approval flow.
Step 5: stateful conversation across turns (one RunContext per session).

Run with:  python -m max_ai.cli.chat
Exit with: /exit, Ctrl+D, or Ctrl+C
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime

from max_ai.base.agent import Agent
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.event_type import (
    ErrorEvent,
    ModelStreamChunkEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)
from max_ai.core.models import ModelConfig
from max_ai.middleware.logging import LoggingMiddleware  # adjust if path differs
from max_ai.tools.function_as_tool import FunctionAsTool  # adjust if path differs
from max_ai.types.agent_response import AgentResponse
from max_ai.types.run_context import RunContext
from max_ai.types.tools import ToolApprovalMode


APP_NAME = "MaxAI"
APP_BANNER = r"""
 __  __              _    ___
|  \/  | __ _ __  __/ \  |_ _|
| |\/| |/ _` |\ \/ / _ \  | |
| |  | | (_| | >  < ___ \ | |
|_|  |_|\__,_|/_/\_\   \_\___|
"""


# ---------------------------------------------------------------------------
# Logging — sent to a file so chat output stays clean.
# Tail in another terminal:   tail -f max_ai.log
# ---------------------------------------------------------------------------
LOG_FILE = os.getenv("LOG_FILE", "max_ai.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    filename=LOG_FILE,
    filemode="w",  # truncate on each run; switch to "a" to keep history
    force=True,  # override anything else that touched the root logger
)

# Silence chatty third-party libraries. Add to the tuple if a new one
# starts spamming. They stay at WARNING, so real errors still surface.
for _noisy in ("httpcore", "httpx", "urllib3", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Hardcoded config — v1 per the spec. Swap to argparse / config files later.
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def build_client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=MODEL,
        host=OLLAMA_HOST,
        config=ModelConfig(
            max_context_window=1000,
            supports_function_calling=True,
            supports_thinking=False,
        ),
        think=False,
    )


# ---------------------------------------------------------------------------
# Demo tools — enough variety to exercise auto-approve and approval-required.
# ---------------------------------------------------------------------------
def build_toolset() -> list[FunctionAsTool]:
    async def get_time() -> str:
        """Return the current local time as an ISO 8601 string."""
        return datetime.now().isoformat(timespec="seconds")

    async def add(a: float, b: float) -> float:
        """Add two numbers and return the sum.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    async def delete_file(path: str) -> str:
        """Pretend to delete a file at the given path.

        This is a stub — it does not actually delete anything. It exists
        to exercise the approval flow.

        Args:
            path: Filesystem path that would be deleted.
        """
        return f"(stub) would have deleted: {path}"

    return [
        FunctionAsTool(get_time, approval_mode=ToolApprovalMode.ASK_APPROVED),
        FunctionAsTool(add, approval_mode=ToolApprovalMode.AUTO_APPROVED),
        FunctionAsTool(delete_file, approval_mode=ToolApprovalMode.ASK_APPROVED),
    ]


def build_agent() -> Agent:
    """Construct the agent used for this REPL session.

    If `Agent` is actually abstract in your tree, replace `Agent(...)`
    below with your concrete subclass.
    """
    return Agent(
        name="chat",
        description="Interactive terminal assistant.",
        instructions=(
            "You are a helpful assistant. Use the available tools when "
            "they help answer the user's question. Be concise."
        ),
        client=build_client(),
        toolset=build_toolset(),
        middlewares=[LoggingMiddleware(level="debug")],
    )


# ---------------------------------------------------------------------------
# Event rendering — used by both initial run and resume.
# ---------------------------------------------------------------------------
class _StreamPrinter:
    """Stateful printer for one turn — handles assistant prefix, newlines."""

    def __init__(self) -> None:
        self._mid_assistant_line = False

    def handle(self, item: object) -> None:
        if isinstance(item, ModelStreamChunkEvent):
            self._handle_chunk(item)
        elif isinstance(item, ToolCallEvent):
            self._break_assistant_line()
            print(f"  → calling {item.tool_name}({_format_params(item.parameters)})")
        elif isinstance(item, ToolCallResponseEvent):
            self._break_assistant_line()
            self._handle_tool_response(item)
        elif isinstance(item, ToolApprovalEvent):
            # Surfaced inline so the user sees it before the prompt.
            self._break_assistant_line()
            reason = (
                f" — {item.reason_for_approval}" if item.reason_for_approval else ""
            )
            print(f"  ⏸ approval needed for {item.tool_name}{reason}")
        elif isinstance(item, ErrorEvent):
            self._break_assistant_line()
            print(f"  ! error ({item.error_type}): {item.error_message}")

    def finish(self) -> None:
        if self._mid_assistant_line:
            print()
            self._mid_assistant_line = False

    # -- internal --
    def _handle_chunk(self, item: ModelStreamChunkEvent) -> None:
        if item.is_final or not item.chunk:
            return
        if not self._mid_assistant_line:
            print("MaxAI> ", end="", flush=True)
            self._mid_assistant_line = True
        print(item.chunk, end="", flush=True)

    def _handle_tool_response(self, item: ToolCallResponseEvent) -> None:
        result = item.tool_result
        if result is None:
            print("  ← (no result)")
            return
        if result.success:
            preview = _truncate(str(result.result), 80)
            print(f"  ← ok: {preview}")
        else:
            preview = _truncate(str(result.error), 80)
            print(f"  ← failed: {preview}")

    def _break_assistant_line(self) -> None:
        if self._mid_assistant_line:
            print()
            self._mid_assistant_line = False


def _format_params(params: dict) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={_truncate(repr(v), 40)}" for k, v in params.items())


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Approval flow.
# ---------------------------------------------------------------------------
def _prompt_approvals(ctx: RunContext) -> bool:
    """Prompt the user for each pending approval.

    Returns True if the run should be resumed (any decision was made),
    False if the user wants to abort the turn entirely.

    Mutates ctx.tool_state via apply_approval.
    """
    pending = list(ctx.tool_state.pending_approvals)
    if not pending:
        return False

    print(f"\n{len(pending)} tool call(s) need approval:")
    bulk_choice: str | None = None  # "all" or "none" once chosen, applies to rest.

    for i, record in enumerate(pending, start=1):
        params = _format_params(getattr(record, "parameters", {}) or {})
        print(f"  [{i}/{len(pending)}] {record.tool_name}({params})")

        if bulk_choice == "all":
            decision = "y"
        elif bulk_choice == "none":
            decision = "n"
        else:
            decision = _ask_decision()

        if decision == "all":
            bulk_choice = "all"
            decision = "y"
        elif decision == "none":
            bulk_choice = "none"
            decision = "n"
        elif decision == "abort":
            print("  (turn aborted — pending approvals left unresolved)")
            return False

        approved = decision == "y"
        ctx.tool_state.apply_approval(record.id, approved=approved)
        print(f"     → {'approved' if approved else 'rejected'}")

    return True


def _ask_decision() -> str:
    """Returns one of: 'y', 'n', 'all', 'none', 'abort'."""
    while True:
        try:
            raw = input("     approve? [y/n/all/none/abort]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "abort"
        if raw in {"y", "yes"}:
            return "y"
        if raw in {"n", "no"}:
            return "n"
        if raw == "all":
            return "all"
        if raw == "none":
            return "none"
        if raw in {"a", "abort", "q"}:
            return "abort"
        print("     (please answer y / n / all / none / abort)")


# ---------------------------------------------------------------------------
# Turn driver — handles initial run + any number of resume cycles.
# ---------------------------------------------------------------------------
async def _drive_turn(agent: Agent, ctx: RunContext, task: str) -> AgentResponse:
    """Run one full turn, looping through approval cycles until the
    agent reaches a real terminal state.
    """
    response = await _stream_once(
        agent.run_stream_events(
            task=task,
            run_context=ctx,
            stream_tokens=True,
        )
    )

    # Resume loop — keep going as long as the agent paused for approvals
    # and the user provided decisions.
    while response.finish_reason == "approval_needed":
        should_resume = _prompt_approvals(ctx)
        if not should_resume:
            break
        response = await _stream_once(
            agent.resume_stream_events(
                run_context=ctx,
                stream_tokens=True,
            )
        )

    return response


async def _stream_once(stream) -> AgentResponse:
    """Consume one event stream, printing as we go, returning the response."""
    printer = _StreamPrinter()
    response: AgentResponse | None = None

    async for item in stream:
        if isinstance(item, AgentResponse):
            response = item
        else:
            printer.handle(item)

    printer.finish()
    assert response is not None, "engine did not yield an AgentResponse"
    return response


# ---------------------------------------------------------------------------
# REPL loop.
# ---------------------------------------------------------------------------
def _print_banner(agent: Agent) -> None:
    line = "=" * 64
    print(line)
    print(APP_BANNER.strip("\n"))
    print(f"{APP_NAME} CLI - interactive agent session")
    print(line)
    print(f"agent: {agent.name}")
    print(f"model: {MODEL}")
    print(f"host:  {OLLAMA_HOST}")
    print(f"logs:  {LOG_FILE}  (tail -f {LOG_FILE})")
    print("commands: /exit, /clear")
    print("Ctrl+D / Ctrl+C also work.")
    print()


async def repl(agent: Agent) -> None:
    # Prepare upfront so config errors surface before the first prompt.
    await agent.prepare()

    # One context for the whole session — stateful conversation comes
    # for free: each turn appends to ctx.messages, the agent sees it all.
    ctx = RunContext()

    _print_banner(agent)

    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n(interrupted)")
            break

        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            break

        if user_input == "/clear":
            ctx = RunContext()
            print("(history cleared)\n")
            continue

        try:
            response = await _drive_turn(agent, ctx, user_input)
        except KeyboardInterrupt:
            print("\n(turn interrupted)")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n! unhandled: {type(exc).__name__}: {exc}")
            continue

        if response.finish_reason not in {"stop", "tool_direct_return"}:
            print(f"  [finish_reason: {response.finish_reason}]")


def main() -> None:
    try:
        agent = build_agent()
    except Exception as exc:  # noqa: BLE001
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(repl(agent))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
