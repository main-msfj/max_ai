"""Expose an Agent as a regular tool for agent composition."""

from __future__ import annotations

import logging
import typing as t
from collections.abc import AsyncGenerator, Callable

from ..base.agent import Agent
from ..base.component import Component
from ..base.tools import CoreTool, ToolContext
from ..core.event_type import CoreEvent
from ..core.messages import CoreMessage
from ..loggers import ScopedLogger
from ..termination import CancellationToken
from ..types.agent_response import AgentResponse
from ..types.run_context import RunContext
from ..types.tool_call import ToolCallRecord, ToolResult
from ..types.tools import AgentAsToolConfig, CoreToolParameters, ToolApprovalMode

logger = logging.getLogger(__name__)
log = ScopedLogger(logger, scope="AgentAsTool")

ResultStrategy = str | Callable[[list[CoreMessage]], str]


class AgentAsTool(Component[AgentAsToolConfig], CoreTool):
    """Wrap an ``Agent`` so another agent can call it as a tool.

    The parent agent sends a string task to this tool. The wrapped child
    agent runs that task and the tool returns text extracted from the
    child response according to ``strategy``:

    - ``"last"``: last message in the child transcript.
    - ``"last:N"``: last N messages joined by newlines.
    - ``"all"``: all child messages joined by newlines.
    - callable: runtime-only custom extractor. Callable strategies are
      intentionally not serializable.
    """

    component_schema = AgentAsToolConfig
    component_type = "tool"

    def __init__(
        self,
        agent: Agent,
        input_name: str = "task",
        strategy: ResultStrategy = "last",
        *,
        name: str | None = None,
        description: str | None = None,
        approval_mode: ToolApprovalMode | str = ToolApprovalMode.AUTO_APPROVED,
        timeout_seconds: float = 300,
        max_retries: int = 0,
    ) -> None:
        if not isinstance(agent, Agent):
            raise TypeError("agent must be an instance of Agent")
        if not isinstance(input_name, str) or not input_name.strip():
            raise ValueError("input_name must be a non-empty string")

        self.agent = agent
        self.input_name = input_name.strip()
        self.strategy = strategy
        self._validate_strategy()

        super().__init__(
            name=name or agent.name,
            description=description or agent.description,
            approval_mode=approval_mode,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    # -------- COMPONENT SERIALIZATION -----------------------------------------------------------
    def _to_config(self) -> AgentAsToolConfig:
        if callable(self.strategy):
            raise TypeError("AgentAsTool callable strategies cannot be serialized.")
        return AgentAsToolConfig(
            agent=self.agent.dump_component().model_dump(exclude_none=True),
            input_name=self.input_name,
            strategy=self.strategy,
        )

    @classmethod
    def _from_config(cls, config: AgentAsToolConfig) -> "AgentAsTool":
        agent = Agent.load_component(config.agent, expected=Agent)
        return cls(agent=agent, input_name=config.input_name, strategy=config.strategy)

    # -------- STRATEGY -----------------------------------------------------------
    def _validate_strategy(self) -> None:
        if callable(self.strategy):
            return
        if not isinstance(self.strategy, str):
            raise AgentAsToolInvalidStrategyError(self.strategy)
        if self.strategy in {"all", "last"}:
            return
        if self.strategy.startswith("last:"):
            _, _, raw_count = self.strategy.partition(":")
            if raw_count.isdigit() and int(raw_count) > 0:
                return
        raise AgentAsToolInvalidStrategyError(self.strategy)

    def _extract_result(self, messages: list[CoreMessage]) -> str:
        if not messages:
            return ""

        if callable(self.strategy):
            result = self.strategy(messages)
            if not isinstance(result, str):
                raise AgentAsToolResultTypeError(result)
            return result

        if self.strategy == "last":
            return messages[-1].text()
        if self.strategy == "all":
            return "\n".join(message.text() for message in messages)
        if self.strategy.startswith("last:"):
            count = int(self.strategy.partition(":")[2])
            return "\n".join(message.text() for message in messages[-count:])

        raise AgentAsToolInvalidStrategyError(self.strategy)

    # -------- TOOL CONTRACT -----------------------------------------------------------
    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                self.input_name: {
                    "type": "string",
                    "description": f"Task for {self.agent.name} to complete.",
                }
            },
            "required": [self.input_name],
            "additionalProperties": False,
        }

    def validate_parameters(self, tool_request: ToolCallRecord) -> CoreToolParameters:
        base = super().validate_parameters(tool_request)
        if not base.is_tool_valid:
            return base

        task = tool_request.parameters.get(self.input_name)
        if not isinstance(task, str) or not task.strip():
            return CoreToolParameters(
                is_tool_valid=False,
                msg_error=f"Parameter '{self.input_name}' must be a non-empty string",
            )
        return base

    def _prepare_execution(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None,
    ) -> tuple[str, RunContext] | ToolResult:
        tool_params = self.validate_parameters(tool_request)
        metadata = {"agent_name": self.agent.name}
        if not tool_params.is_tool_valid:
            return ToolResult.invalid_parameters(
                tool_request.id,
                tool_params.msg_error or "Invalid parameters",
            )

        task = t.cast(str, tool_request.parameters[self.input_name]).strip()
        ctx = RunContext(
            user_id=tool_context.user_id if tool_context is not None else "user_001",
            session_id=tool_context.session_id if tool_context is not None else None,
        )
        ctx.runtime_state.shared_state["parent_tool_call_id"] = tool_request.id
        ctx.runtime_state.shared_state["parent_agent_tool"] = metadata
        return task, ctx

    def _build_tool_result(
        self,
        tool_request: ToolCallRecord,
        response: AgentResponse | None,
    ) -> ToolResult:
        metadata: dict[str, t.Any] = {"agent_name": self.agent.name}
        if response is None:
            return ToolResult.tool_failure(tool_request.id, "Agent returned no response.", metadata=metadata)
        if not response.messages:
            return ToolResult.tool_failure(tool_request.id, "Agent returned no messages.", metadata=metadata)

        content = self._extract_result(response.messages)
        metadata.update(
            message_count=len(response.messages),
            finish_reason=response.finish_reason,
            usage=response.usage.model_dump(),
        )
        return ToolResult.success_result(tool_request.id, content, metadata=metadata)

    async def execute(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        prepared = self._prepare_execution(tool_request, tool_context)
        if isinstance(prepared, ToolResult):
            return prepared
        task, ctx = prepared

        try:
            response = await self.agent.run(
                task=task,
                run_context=ctx,
                cancellation_token=cancellation_token,
                stream_tokens=False,
            )
            return self._build_tool_result(tool_request, response)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Agent execution failed: {exc}"
            log.error(error_msg, agent_name=self.agent.name)
            return ToolResult.execution_error(tool_request.id, error_msg)

    async def execute_stream(
        self,
        tool_request: ToolCallRecord,
        tool_context: ToolContext | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[CoreEvent | ToolResult, None]:
        prepared = self._prepare_execution(tool_request, tool_context)
        if isinstance(prepared, ToolResult):
            yield prepared
            return
        task, ctx = prepared

        response: AgentResponse | None = None
        try:
            async for item in self.agent.run_stream_events(
                task=task,
                run_context=ctx,
                cancellation_token=cancellation_token,
                stream_tokens=False,
            ):
                if isinstance(item, AgentResponse):
                    response = item
                else:
                    yield item
        except Exception as exc:  # noqa: BLE001
            yield ToolResult.execution_error(tool_request.id, f"Agent execution failed: {exc}")
            return

        yield self._build_tool_result(tool_request, response)


class AgentAsToolResultTypeError(Exception):
    """Raised when a custom extraction strategy returns a non-string value."""

    def __init__(self, result: t.Any) -> None:
        super().__init__(
            f"AgentAsToolResultTypeError: invalid result {result!r}. "
            "Expected a string."
        )


class AgentAsToolInvalidStrategyError(Exception):
    """Raised when the result extraction strategy is invalid."""

    def __init__(self, strategy: t.Any) -> None:
        super().__init__(
            f"AgentAsToolInvalidStrategyError: invalid strategy {strategy!r}. "
            "Expected 'last', 'all', 'last:N', or a callable."
        )
