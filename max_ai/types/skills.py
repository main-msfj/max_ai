import typing as t
from pathlib import Path
from pydantic import BaseModel, Field

from ..core.blocks import SkillBlock


# -------- SKILLS -----------------------------------------------------------
class ResourceMeta(BaseModel):
    """Metadata for an auxiliary resource file inside a skill package.

    Resources are the ``.md`` files a skill ships alongside its
    SKILL.md — style guides, examples, reference material. They are
    exposed to the LLM on demand via the ``read_skill_resource`` tool
    rather than injected into the prompt upfront.
    """

    filename: str = Field(..., description="Resource filename (e.g. 'style_guide.md')")
    path: Path = Field(..., description="Absolute path to the resource file")
    description: str = Field(..., description="What the resource contains")

class Skill(BaseModel):
    """A fully-loaded skill ready to be plugged into an agent.

    Produced by a ``CoreSkillRegistry.load()`` call. Carries everything
    the agent needs: the prompt block to inject, the tools discovered
    from its scripts, and the resources catalog for on-demand reads.
    """

    model_config = {"arbitrary_types_allowed": True}

    block: SkillBlock = Field(..., description="Prompt block for the SkillsLayer")
    tools: list[t.Any] = Field(  # type: ignore 
        default_factory=list,
        description="Tools auto-wrapped from the skill's script functions",
    )
    resources: dict[str, ResourceMeta] = Field(
        default_factory=dict,
        description="Auxiliary resources keyed by filename",
    )

    @property
    def name(self) -> str:
        return self.block.name