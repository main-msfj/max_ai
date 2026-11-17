"""What is stored of a summary compaction."""

from ....base.compaction import WindowCompactionConfig
from ....base.component import ComponentModel


class SummaryCompactionConfig(WindowCompactionConfig):
    instructions: str = ""  # what to care about in the summary, added to the task
    client: ComponentModel | None = None  # the model that summarizes; None = the agent's
