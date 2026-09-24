"""Store a local workspace in another folder that represents a backend."""

import asyncio
import shutil
from pathlib import Path

from max_ai.base.workspace import WorkspaceBase


class FolderWorkspace(WorkspaceBase):
    def __init__(self, root: Path, backend: Path) -> None:
        super().__init__(root=root)
        self.backend = backend

    async def download(self, user_id: str, conversation_id: str | None = None) -> None:
        local = self.materialize(user_id, conversation_id).workspace_dir
        remote = self.backend / user_id
        if remote.exists():
            shutil.copytree(remote, local, dirs_exist_ok=True)

    async def upload(self, user_id: str, conversation_id: str | None = None) -> None:
        local = self.materialize(user_id, conversation_id).workspace_dir
        shutil.copytree(local, self.backend / user_id, dirs_exist_ok=True)


async def main() -> None:
    workspace = FolderWorkspace(Path("./local/agents"), Path("./local/backend"))
    folder = workspace.materialize("user_ana").workspace_dir
    (folder / "note.txt").write_text("A note stored in the workspace.", encoding="utf-8")
    await workspace.upload("user_ana")
    print(Path("./local/backend/user_ana/note.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
