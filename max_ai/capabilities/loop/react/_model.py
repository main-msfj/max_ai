"""What is stored of a ReAct loop."""

from pydantic import BaseModel, Field

from ....base.component import ComponentModel


class ReactLoopConfig(BaseModel):
    max_iterations: int = Field(default=50, gt=0)  # model calls allowed in one turn
    compaction: ComponentModel | None = None  # None = the loop's own, that drops old messages
