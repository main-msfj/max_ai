"""Terminal chat interface for max_ai agents.

A REPL (read-eval-print loop) that drives an ``Agent`` from the terminal:
type a message, watch the agent stream its answer, see its plan, tool calls
and reasoning live, and answer the agent's own questions inline. The whole
thing is a thin consumer of ``agent.run_stream_events`` — the same event
stream the web UI consumes, rendered with Rich instead of SSE.

Usage::

    from max_ai.cli import run_repl
    await run_repl(agent)

See ``examples/agent_cli.py`` for a runnable entry point.
"""

from .repl import run_repl
from .renderer import CliRenderer

__all__ = ["run_repl", "CliRenderer"]
