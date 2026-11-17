"""MaxAI web UI: the same agent as the CLI, in the browser, with a Trajectory
tab that shows each turn's model calls and tools on a timeline.

```python
from max_ai.web import run_web

await run_web(agent, store=LocalSessionStore("./sessions"), user_id="user_001")
```
"""

from .server import WebHost, run_web

__all__ = ["WebHost", "run_web"]
