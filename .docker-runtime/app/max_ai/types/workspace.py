from pathlib import Path
from pydantic import BaseModel, Field


# -------- WORKSPACE -----------------------------------------------------------
class WorkspaceDirectory(BaseModel):
    """Workspace Containers"""
    root: Path = Field(..., description="Root DIR")
    tool_dir: Path = Field(..., description="Tools DIR")
    skill_dir: Path = Field(..., description="SKills DIR")
    artifacts_dir: Path = Field(..., description="Generated artifacts DIR")
