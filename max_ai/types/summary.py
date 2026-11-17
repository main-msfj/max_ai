"""What is kept of a conversation when its old messages leave."""

from pydantic import BaseModel, Field


class Summary(BaseModel):
    """The old part of a conversation, in fixed sections."""

    goal: str = Field(default="", description="What the user wants, in their words")
    decisions: list[str] = Field(default_factory=list[str], description="What was decided, and why")
    done: list[str] = Field(default_factory=list[str], description="What is finished")
    pending: list[str] = Field(default_factory=list[str], description="What is left, in order")
    files: list[str] = Field(
        default_factory=list[str], description="Files created or changed, one line each"
    )
    facts: list[str] = Field(
        default_factory=list[str],
        description="Things learned that still matter: errors found, commands that work",
    )
    last_request: str = Field(default="", description="The user's most recent request, as it was")

    def text(self) -> str:
        """The summary as the model reads it: one title per section that has something."""
        parts: list[str] = []
        for name, value in self:
            if not value:
                continue
            title = name.replace("_", " ").capitalize()
            body = value if isinstance(value, str) else "\n".join(f"- {item}" for item in value)
            parts.append(f"{title}:\n{body}")
        return "\n\n".join(parts)
