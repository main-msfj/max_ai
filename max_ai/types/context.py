"""How the model's window is shared out in a call."""

from pydantic import BaseModel


class Context(BaseModel):
    """The window in tokens, part by part: together they are the window."""

    window: int = 0  # the model's whole window
    prompt: int = 0  # the system prompt, with the summary after a compaction
    tools: int = 0  # the definitions of the loaded tools
    messages: int = 0  # the conversation: the only part that grows
    output: int = 0  # kept for the model's answer
    margin: int = 0  # never used: token counts are not exact

    @property
    def capacity(self) -> int:
        """Tokens the messages may take before a compaction."""
        return max(0, self.window - self.prompt - self.tools - self.output - self.margin)

    @property
    def free(self) -> int:
        """Tokens the messages can still grow."""
        return max(0, self.capacity - self.messages)
