"""What is stored of a ReAct loop."""

from pydantic import BaseModel, Field


class ReactLoopConfig(BaseModel):
    max_iterations: int = Field(default=50, gt=0)  # model calls allowed in one turn
