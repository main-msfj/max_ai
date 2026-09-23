from pathlib import Path

from pydantic import BaseModel, Field


# -------- WORKSPACE -----------------------------------------------------------
class WorkspaceDirectory(BaseModel):
    """Workspace Containers"""
    root: Path = Field(..., description="Root DIR")
    workspace_dir: Path = Field(..., description="The user's single shared project workspace")
    scratch_dir: Path | None = Field(default=None, description="Scratchpad dir for this conversation")
    skill_dir: Path = Field(..., description="SKills DIR")
    artifacts_dir: Path = Field(..., description="Generated artifacts DIR")
