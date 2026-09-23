import pytest

from max_ai.core.stacks.container import LayerContainer
from max_ai.types.stacks import PromptCtx


@pytest.fixture
def prompts() -> PromptCtx:
    return PromptCtx(stack=LayerContainer([]))
