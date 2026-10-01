from pathlib import Path

from pydantic import BaseModel, Field


# -------- WORKSPACE -----------------------------------------------------------
class WorkspaceDirectory(BaseModel):
    """Workspace Containers"""
    root: Path = Field(..., description="Root DIR")
    workspace_dir: Path = Field(..., description="The user's single shared project workspace")
    skill_dir: Path = Field(..., description="SKills DIR")
    artifacts_dir: Path = Field(..., description="Generated artifacts DIR")
