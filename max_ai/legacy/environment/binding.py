"""Internal bridge for tools that consume the existing ToolContext.environment."""

from .environment import Environment
from ...base.executor import ExecutorBase, ExecutionSession


class SessionEnvironment(Environment):
    def __init__(self, executor: ExecutorBase, session: ExecutionSession):
        super().__init__(session.workspace, session.user_id, session.conversation_id)
        self.executor = executor
        self.session = session

    @property
    def variables(self):
        variables = {"WORKSPACE": self.session.workspace_path}
        scratch = getattr(self.session.handle, "scratch_dir", None)
        if scratch is not None:
            variables["SCRATCHPAD"] = str(scratch)
        return variables

    async def start(self):
        # The manager already connected this session.
        pass

    async def execute(self, command, *, timeout=60, cancellation_token=None):
        return await self.executor.execute(
            self.session, command, timeout=timeout, cancellation_token=cancellation_token,
        )

    async def stop(self):
        # Provider execute owns process cancellation. The manager owns session
        # cleanup, after remote files have been synchronized.
        pass
