"""The bash tool: the model sends a command, the harness runs it in the executor."""

from ._permissions import command_matches, split_command
from ._tool import BashTool, build_script

__all__ = ["BashTool", "build_script", "command_matches", "split_command"]
