"""LocalSandbox: this computer is the sandbox."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ....base.sandbox import SandboxCore


class LocalSandboxConfig(BaseModel):
    root: str = "."


class LocalSandbox(SandboxCore):
    """This computer is the sandbox. ``root`` is where ``.agent/`` goes;
    ``.`` is the folder the program runs in."""

    component_schema = LocalSandboxConfig
    component_provider_override = "max_ai.capabilities.sandbox.local.LocalSandbox"

    def __init__(self, root: str = ".") -> None:
        super().__init__(project=Path(root))
        self.create_project()

    def _to_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(root=str(self.root.parent))

    @classmethod
    def _from_config(cls, config: LocalSandboxConfig) -> LocalSandbox:
        return cls(root=config.root)

    async def synchronize_environment(self, skill_var: dict[str, str], identifier: str) -> None:
        raise NotImplementedError

    async def upload_file(self, path: str, content: bytes) -> None:
        raise NotImplementedError

    async def get_or_update_skill(self, name: str) -> None:
        raise NotImplementedError

