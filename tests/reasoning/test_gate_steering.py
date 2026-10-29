"""A gate rejection steers the next model call only; it is never persisted."""

from types import SimpleNamespace

from max_ai.agents.agent import Agent
from max_ai.base.completion_gate import CompletionBase, CompletionDecision
from max_ai.capabilities.workspace.local import LocalWorkspace
from max_ai.core.messages import HARNESS_SOURCE, AssistantMessage, SystemMessage
from max_ai.types.completions import ChatCompletionResult, Usage
from max_ai.types.run_context import RunContext


class SeeingClient:
    """Answers every call and records what the model was shown."""

    model = "fake"
    config = SimpleNamespace(tokenizer_base="o200k_base")

    def __init__(self):
        self.seen = []

    async def run(self, ctx, prompts, tools=None, **kwargs):
        self.seen.append(list(ctx.messages))
        return ChatCompletionResult(
            message=AssistantMessage(source="fake", content=f"answer {len(self.seen)}"),
            usage=Usage(), model=self.model, finish_reason="stop",
        )


class RejectsOnce(CompletionBase):
    def __init__(self):
        super().__init__()
        self.rejected = False

    def on_final_response(self, ctx):
        if self.rejected:
            return CompletionDecision(status="completed")
        self.rejected = True
        return CompletionDecision(status="incomplete", reasons=("Check the totals first.",))


def harness_notes(messages):
    return [m.text() for m in messages
            if isinstance(m, SystemMessage) and m.source == HARNESS_SOURCE]


async def test_a_rejection_reaches_the_next_call_but_not_the_transcript(tmp_path):
    client = SeeingClient()
    agent = Agent(name="t", description="d", instructions="i", client=client,
                  workspace=LocalWorkspace(root=tmp_path), gates=[RejectsOnce()])
    async with agent:
        ctx = RunContext(user_id="u", session_id="s")
        await agent.run("sum it", run_context=ctx)

    assert len(client.seen) == 2
    assert harness_notes(client.seen[0]) == []
    assert harness_notes(client.seen[1]) == ["Check the totals first."]
    assert harness_notes(ctx.messages) == []
    assert [m.text() for m in ctx.messages if isinstance(m, AssistantMessage)] == [
        "answer 1", "answer 2",
    ]
