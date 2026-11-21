"""The rules of our harness: what an agent must have and what we add to it."""

import logging
import os
import typing as t
from collections.abc import Sequence

from ...base.compaction import CoreCompaction
from ...base.layer import CoreLayer
from ...base.middleware import CoreMiddleware
from ...base.sandbox import SandboxCore
from ...base.skills import SkillsCore
from ...base.tools import CoreTool
from ...config import setting
from ...errors.harness import HarnessError
from ..model.llm import ModelConfig
from ..retry import RetryPolicy
from ..tokenizer import TokenCounter

if t.TYPE_CHECKING:
    from ...capabilities.loop import ReactLoop
    from ...capabilities.memory import Memory
    from ..tool.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Harness:
    """Rules as class methods. Some check and raise ``HarnessError``; the
    others return what the harness adds (tools, middlewares, compaction)."""

    BASH: t.ClassVar[str] = "bash"

    TOOL_TOO_LONG: t.ClassVar[str] = (
        "The description of tool {name!r} takes about {tokens} tokens; the most is {limit}. "
        "Say what it does and when to use it, and move the rest to its parameters."
    )
    TOOLS_TOO_MANY: t.ClassVar[str] = (
        "The loaded tools take about {tokens} tokens; the most is {limit}. "
        "Pass fewer tools, or mark some as deferred so they are found with search_tools."
    )
    PROMPT_SHORT: t.ClassVar[str] = (
        "The system prompt takes about %s tokens; under %s the provider's cache is not used well."
    )
    PROMPT_TOO_LONG: t.ClassVar[str] = (
        "The system prompt takes about {tokens} tokens; the most is {limit} "
        "({share:.0%} of the model's window of {window})."
    )
    INSTRUCTIONS_TOO_LONG: t.ClassVar[str] = (
        "The instructions take about {tokens} tokens; the most is {limit}. "
        "Say in a few lines how the agent behaves; details go in a prompt layer or a skill."
    )
    CUSTOM_PROMPT_TOO_LONG: t.ClassVar[str] = (
        "Your prompt layers take about {tokens} tokens; the most is {limit}. "
        "Write a skill with the details instead: it is read only when it is needed."
    )
    WINDOW_OVERWRITTEN: t.ClassVar[str] = (
        "The harness sets the model's limits: context window %s (was %s), output %s (was %s)."
    )
    BASH_NEEDS_SANDBOX: t.ClassVar[str] = (
        "bash runs its commands in a sandbox: pass sandbox=... to the agent."
    )
    DESCRIPTION_TOO_LONG: t.ClassVar[str] = (
        "The description takes about {tokens} tokens; the most is {limit}. "
        "Say in a line or two what the agent is for."
    )
    TOOL_OUTLIVES_SANDBOX: t.ClassVar[str] = (
        "A tool may run {tool:.0f} seconds (tool_timeout_seconds) but a sandbox lives "
        "{sandbox:.0f} (harness_sandbox_seconds); lower the first or raise the second."
    )
    MEMORY_NEEDS_SANDBOX: t.ClassVar[str] = (
        "The memory is kept in a sandbox: pass sandbox=... to the agent."
    )
    SKILLS_NEED_SANDBOX: t.ClassVar[str] = (
        "Skills are read and run in a sandbox: pass sandbox=... to the agent."
    )
    SKILLS_NEED_BASH: t.ClassVar[str] = (
        "Skills run their scripts with bash: add BashTool() to the toolset."
    )

    # -------- RULES THAT CHECK ---------------------------------------------------
    @classmethod
    def tool_rules(cls, tools: Sequence[CoreTool], counter: TokenCounter | None = None) -> None:
        """The developer's tools: each description stays under the limit."""
        limit = setting.harness_tool_description_max_tokens
        for tool in tools:
            tokens = TokenCounter.estimate(tool.description, counter)
            if tokens > limit:
                text = cls.TOOL_TOO_LONG.format(name=tool.name, tokens=tokens, limit=limit)
                raise HarnessError(text)

    @classmethod
    def tools_total_rules(cls, tool_tokens: int) -> None:
        """All the loaded tools together stay under the limit."""
        limit = setting.harness_tools_max_tokens
        if tool_tokens > limit:
            raise HarnessError(cls.TOOLS_TOO_MANY.format(tokens=tool_tokens, limit=limit))

    @classmethod
    def prompt_rules(cls, prompt_tokens: int, window: int) -> None:
        """The whole system prompt: a warning when short, an error when too long."""
        # 1. Too short only costs money: say it and go on.
        if prompt_tokens < setting.harness_prompt_min_tokens:
            logger.warning(cls.PROMPT_SHORT, prompt_tokens, setting.harness_prompt_min_tokens)
        # 2. Too long leaves no room for the conversation.
        share = setting.harness_prompt_max_share
        limit = int(window * share)
        if prompt_tokens > limit:
            raise HarnessError(
                cls.PROMPT_TOO_LONG.format(
                    tokens=prompt_tokens, limit=limit, share=share, window=window
                )
            )

    @classmethod
    def instruction_rules(cls, instructions: str, counter: TokenCounter | None = None) -> None:
        """How the agent behaves, in a few lines."""
        limit = setting.harness_instructions_max_tokens
        tokens = TokenCounter.estimate(instructions, counter)
        if tokens > limit:
            raise HarnessError(cls.INSTRUCTIONS_TOO_LONG.format(tokens=tokens, limit=limit))

    @classmethod
    def description_rules(cls, description: str, counter: TokenCounter | None = None) -> None:
        """The agent's description is a line or two."""
        limit = setting.harness_description_max_tokens
        tokens = TokenCounter.estimate(description, counter)
        if tokens > limit:
            raise HarnessError(cls.DESCRIPTION_TOO_LONG.format(tokens=tokens, limit=limit))

    @classmethod
    def custom_prompt_rules(cls, layers: Sequence[CoreLayer], counter: TokenCounter | None = None) -> None:
        """The developer's own prompt layers, together; more than this belongs in a skill."""
        limit = setting.harness_custom_prompt_max_tokens
        tokens = sum(TokenCounter.estimate(layer.render({}), counter) for layer in layers)
        if tokens > limit:
            raise HarnessError(cls.CUSTOM_PROMPT_TOO_LONG.format(tokens=tokens, limit=limit))

    @classmethod
    def sandbox_rules(cls) -> float:
        """Seconds a sandbox lives; no tool may be allowed to run longer."""
        tool, sandbox = setting.tool_timeout_seconds, setting.harness_sandbox_seconds
        if tool > sandbox:
            raise HarnessError(cls.TOOL_OUTLIVES_SANDBOX.format(tool=tool, sandbox=sandbox))
        return sandbox

    # -------- RULES THAT COMPLETE ------------------------------------------------
    @classmethod
    def combination_rules(
        cls,
        toolset: Sequence[CoreTool],
        sandbox: SandboxCore | None,
        skills: SkillsCore | None,
        memory: "Memory | None" = None,
    ) -> None:
        """Commands, skills and the memory live in a sandbox; skills also need bash."""
        names = {tool.name for tool in toolset}
        if memory is not None and sandbox is None:
            raise HarnessError(cls.MEMORY_NEEDS_SANDBOX)
        if skills is not None and sandbox is None:
            raise HarnessError(cls.SKILLS_NEED_SANDBOX)
        if cls.BASH in names and sandbox is None:
            raise HarnessError(cls.BASH_NEEDS_SANDBOX)
        if skills is not None and cls.BASH not in names:
            raise HarnessError(cls.SKILLS_NEED_BASH)

    @classmethod
    def tool_combination(
        cls,
        toolset: Sequence[CoreTool],
        sandbox: SandboxCore | None,
        skills: SkillsCore | None,
        memory: "Memory | None" = None,
    ) -> list[CoreTool]:
        """The tools the agent ends with: the developer's first, then ours.
        ``combination_rules`` already said the pieces go together."""
        from ...capabilities.tools.ask_user import AskUserTool
        from ...capabilities.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
        from ...capabilities.tools.memory import SaveMemoryTool
        from ...capabilities.tools.skill import ReadSkillTool
        from ...capabilities.tools.todos import WriteTodosTool
        from ...capabilities.tools.worktree import EnterWorktreeTool, ExitWorktreeTool

        tools = list(toolset)
        names = {tool.name for tool in tools}

        # 1. Ours: files only with a sandbox; the question and the list always.
        ours: list[CoreTool] = [
            AskUserTool(), WriteTodosTool(approval=setting.harness_plan_approval),
        ]
        if sandbox is not None:
            ours += [ReadFileTool(), WriteFileTool(), EditFileTool()]

        # 2. A memory that may be written has its own tool to save a fact.
        if memory is not None and not memory.read_only:
            ours.append(SaveMemoryTool())

        # 3. An agent with skills reads their instructions with its own tool.
        if skills is not None:
            ours.append(ReadSkillTool())

        # 4. Whoever runs commands can also work on a separate copy of the repository.
        if cls.BASH in names:
            ours += [EnterWorktreeTool(), ExitWorktreeTool()]

        # 5. A tool of the developer with the same name wins.
        return tools + [tool for tool in ours if tool.name not in names]

    @classmethod
    def deferred_rules(cls, registry: "ToolRegistry") -> None:
        """Few deferred tools go loaded: the two tools that find and call them
        would cost more than they save. Many are reached through those two."""
        from ...capabilities.tools.call_tool import CallToolTool
        from ...capabilities.tools.search_tool import SearchToolsTool

        if registry.deferred_tokens() < setting.harness_deferred_tools_min_tokens:
            registry.load_deferred()
        if registry.deferred_tools():
            for extra in (SearchToolsTool(registry), CallToolTool(registry)):
                if registry.get(extra.name) is None:
                    registry.register(extra)

    @classmethod
    def middleware_rules(
        cls, middlewares: Sequence[CoreMiddleware], agent: str = "agent"
    ) -> list[CoreMiddleware]:
        """Tracing is always there, first: the one given, or ours. It names
        its traces after ``agent``."""
        from ...capabilities.middleware.tracing import TracingMiddleware

        given = list(middlewares)
        tracing = next((item for item in given if isinstance(item, TracingMiddleware)), None)
        if tracing is None:
            tracing = TracingMiddleware()
        tracing.agent = agent
        return [tracing, *(item for item in given if item is not tracing)]

    @classmethod
    def loop_rules(cls) -> "ReactLoop":
        """The loop of an agent that was given none: ``harness_max_iterations``
        model calls a turn, and a summary of the messages that leave."""
        from ...capabilities.loop import ReactLoop

        return ReactLoop(
            max_iterations=setting.harness_max_iterations, compaction=cls.compaction_rules()
        )

    @classmethod
    def compaction_rules(cls) -> CoreCompaction[t.Any]:
        """A summary written by ``setting.summary_model`` through OpenRouter,
        whatever model the agent uses; by the agent's own model when no model
        is set or there is no key."""
        from ...capabilities.clients.openrouter import OpenRouterChatCompletionClient
        from ...capabilities.compaction import SummaryCompaction

        # 1. The summary model of the settings, when it has its key.
        key = os.environ.get(OpenRouterChatCompletionClient.API_KEY_ENV or "")
        client = None
        if setting.summary_model and key:
            client = OpenRouterChatCompletionClient(setting.summary_model)

        # 2. The limits are the settings'.
        return SummaryCompaction(
            client=client,
            keep_groups=setting.compaction_keep_groups,
            max_result_chars=setting.compaction_result_chars,
            max_share=setting.compaction_summary_max_share,
        )

    @classmethod
    def retry_rules(cls, given: RetryPolicy) -> RetryPolicy:
        """A model call is tried at most ``harness_retry_attempts`` times; the
        waits of the client's own policy are kept."""
        return given.model_copy(update={"attempts": setting.harness_retry_attempts})

    @classmethod
    def model_rules(cls, config: ModelConfig) -> None:
        """The window and the output are the harness's: what was given is replaced."""
        ours = ModelConfig()
        given = (config.max_context_window, config.max_output_tokens)
        if given != (ours.max_context_window, ours.max_output_tokens):
            logger.warning(
                cls.WINDOW_OVERWRITTEN,
                ours.max_context_window, given[0], ours.max_output_tokens, given[1],
            )
            config.max_context_window = ours.max_context_window
            config.max_output_tokens = ours.max_output_tokens
