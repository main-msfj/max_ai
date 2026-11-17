"""ReAct: the model answers or asks for tools; the tools run; repeat."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel

from ....base.loop import AnyClient, CoreLoop, LoopHook, LoopState, Stop
from ....core.event_type import (
    CoreEvent,
    FatalErrorEvent,
    ReasoningCompleteEvent,
    ReasoningIterationEvent,
)
from ....core.retry import RetryPolicy
from ....core.tool import ToolDispatcher
from ....types.run_context import RunContext
from ....types.tool_call import ToolCallRecord
from ._model import ReactLoopConfig


class ReactLoop(CoreLoop[ReactLoopConfig]):
    """Calls the model until it answers without asking for tools.

    The loop only moves the turn; what to do at each moment is asked to the
    ``hook``. ``state.finish_reason`` says why the turn ended: the model's
    own reason (``stop``, ``length``), ``waiting_for_user``, ``cancelled``,
    ``error``, or the reason of the hook's ``Stop`` (``max_iterations``,
    ``empty_answer``, ``repetition``, ``content_filter``...)."""

    component_schema = ReactLoopConfig
    component_provider_override = "max_ai.capabilities.loop.react.ReactLoop"

    WAITING: t.ClassVar[str] = "waiting_for_user"
    LIMIT: t.ClassVar[str] = "max_iterations"

    def __init__(self, max_iterations: int = 50) -> None:
        self.max_iterations = max_iterations  # model calls allowed in one turn

    def _to_config(self) -> ReactLoopConfig:
        return ReactLoopConfig(max_iterations=self.max_iterations)

    @classmethod
    def _from_config(cls, config: ReactLoopConfig) -> t.Self:
        return cls(max_iterations=config.max_iterations)

    async def run(
        self,
        ctx: RunContext,
        prompt: str,
        client: AnyClient,
        dispatcher: ToolDispatcher,
        *,
        hook: LoopHook | None = None,
        state: LoopState | None = None,
        output_format: type[BaseModel] | None = None,
        stream: bool = False,
        retry: RetryPolicy | None = None,
        max_iterations: int | None = None,  # given here, it wins over the loop's own
    ) -> t.AsyncGenerator[CoreEvent]:
        hook, state = hook or LoopHook(), state or LoopState()
        state.limit = max_iterations or self.max_iterations
        answered = False  # the model ended the turn with an answer of its own
        try:
            # 1. Calls an earlier turn left open: closed if the user wrote
            #    instead of answering, run if approved or answered since.
            self.settle(ctx)
            records: list[ToolCallRecord] = self.open_records(ctx)
            if records:
                async for event in self.run_tools(ctx, dispatcher, records):
                    yield event

            while True:
                # 2. A tool waits for a person: the turn ends until they answer.
                #    If they wrote something else meanwhile, it is closed and the turn goes on.
                if self.waiting(records):
                    if not ctx.inbox:
                        state.finish_reason = self.WAITING
                        break
                    for record in records:
                        if not record.is_consumed:
                            self.close(ctx, record, self.NOT_ANSWERED)
                if records:
                    await hook.after_tools(ctx, state)

                # 3. Before the model: the hook may end the turn or tell it something.
                decision = await hook.before_model(ctx, state)
                if isinstance(decision, Stop):
                    yield self.end(ctx, state, decision)
                    break
                if isinstance(decision, str):
                    yield self.note(ctx, state, decision)

                # 4. The loop's own stop, for a hook that did not end the turn at its limit.
                if state.iteration >= state.limit + self.EXTRA_CALLS:
                    state.finish_reason = self.LIMIT
                    break

                # 5. The model answers, or asks for tools.
                state.iteration += 1
                yield ReasoningIterationEvent(
                    source=self.SOURCE, iteration=state.iteration, max_iterations=state.limit
                )
                async for event in self.ask_model(
                    ctx, prompt, client, dispatcher, state, hook, stream=stream, retry=retry,
                ):
                    yield event
                message = state.message
                assert message is not None
                self.repeated(state, message)
                records = self.new_records(ctx, message)

                # 6. After the model: the hook says how the turn goes on.
                decision = await hook.after_model(ctx, state)
                if isinstance(decision, Stop):
                    for record in records:
                        self.close(ctx, record, self.TURN_ENDED)
                    yield self.end(ctx, state, decision)
                    break

                # 7. No tools: the turn ends, unless the hook had something to say.
                if not records:
                    if decision is None:
                        answered = True
                        break
                    if isinstance(decision, str):
                        yield self.note(ctx, state, decision)
                    continue

                # 8. Tools: answered with the hook's text, or run.
                if isinstance(decision, str):
                    for record in records:
                        self.close(ctx, record, decision)
                    yield self.notice(state, decision)
                    continue
                async for event in self.run_tools(ctx, dispatcher, records):
                    yield event

            # 9. The final answer, in the format asked: one call more, without tools.
            if answered and output_format is not None:
                async for event in self.format_answer(
                    ctx, prompt, client, state, output_format, retry
                ):
                    yield event
        except Exception as error:
            # 10. The turn failed: the hook may answer for the agent, or the error goes up.
            stop = await hook.on_error(ctx, state, error)
            if stop is None:
                state.finish_reason = "error"
                yield FatalErrorEvent(
                    source=self.SOURCE, error_message=str(error), error_type=type(error).__name__
                )
                raise
            yield self.end(ctx, state, stop)
        except BaseException:
            state.finish_reason = "cancelled"
            raise

        yield ReasoningCompleteEvent(
            source=self.SOURCE, finish_reason=state.finish_reason,
            total_iterations=state.iteration,
        )
