"""Who decides if a tool call runs: the user's rules in ``setting.json`` and
the commands the harness never lets run without asking."""

import fnmatch
import re
import typing as t
from collections.abc import Mapping


class Permissions:
    """Rules are ``tool`` or ``tool(pattern)``; the pattern is matched with
    the bash command or the path of a file tool. ``decide`` answers False
    (denied), None (ask the user), True (run) or ``DEFAULT`` (no rule)."""

    DEFAULT: t.ClassVar[str] = "default"
    KINDS: t.ClassVar[tuple[str, ...]] = ("allow", "ask", "deny")
    BASH: t.ClassVar[str] = "bash"
    SETTINGS: t.ClassVar[str] = "setting.json"
    # Always asked, whatever the tool or the user's allow list say.
    DANGEROUS: t.ClassVar[tuple[str, ...]] = (
        r"^rm\b", r"^sudo\b", r"^dd\b", r"^mkfs", r"^shutdown\b", r"^reboot\b",
        r"^(chmod|chown)\s+(-\w*R|--recursive)", r"^git\s+push\b.*(--force|-f\b)",
        r"^git\s+reset\s+--hard", r"^git\s+clean\b", r"^kill(all)?\b", r"^eval\b",
        r"^(sh|bash|zsh|python3?)$",  # something is piped into an interpreter
    )
    # What cannot be read safely: a command built while it runs.
    UNREADABLE: t.ClassVar[tuple[str, ...]] = ("$(", "`", "<(")
    SPLIT: t.ClassVar[str] = r"&&|\|\||;|\||\n"

    def __init__(self, rules: Mapping[str, list[str]] | None = None) -> None:
        given = rules or {}
        self.allow = list(given.get("allow", []))
        self.ask = list(given.get("ask", []))
        self.deny = list(given.get("deny", []))

    def decide(self, tool_name: str, parameters: Mapping[str, t.Any]) -> bool | str | None:
        """What happens with this call; the strictest rule that matches wins."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        # 1. The user's deny list.
        if self.matches(self.deny, tool_name, target):
            return False
        # 2. What the harness always asks, and the user's ask list.
        if self.dangerous(tool_name, target) or self.matches(self.ask, tool_name, target):
            return None
        # 3. The user's allow list; else the tool's own way.
        if self.matches(self.allow, tool_name, target):
            return True
        return self.DEFAULT

    @staticmethod
    def dangerous(tool_name: str, target: str) -> bool:
        """A call the user must see: it touches the settings, or it is a
        command that deletes, takes control or cannot be read safely."""
        if Permissions.SETTINGS in target:
            return True
        if tool_name != Permissions.BASH:
            return False
        if any(mark in target for mark in Permissions.UNREADABLE):
            return True
        # Each part of the command is looked at alone: ``ls && rm x`` has an rm.
        for part in re.split(Permissions.SPLIT, target):
            words = part.strip()
            if any(re.search(pattern, words) for pattern in Permissions.DANGEROUS):
                return True
        return False

    @staticmethod
    def matches(rules: list[str], tool_name: str, target: str) -> bool:
        """Whether one of ``rules`` names this tool and, when it has a pattern, this target."""
        for rule in rules:
            name, _, pattern = rule.partition("(")
            if name.strip() != tool_name:
                continue
            if not pattern or fnmatch.fnmatchcase(target.strip(), pattern.rstrip(")")):
                return True
        return False

    @staticmethod
    def rule_for(tool_name: str, parameters: Mapping[str, t.Any]) -> str:
        """The rule "always allow" keeps: a command by its first word, any
        other tool whole."""
        words = str(parameters.get("command") or "").split()
        if tool_name == Permissions.BASH and words:
            return f"{tool_name}({words[0]}*)"
        return tool_name

    @staticmethod
    def can_be_kept(tool_name: str, parameters: Mapping[str, t.Any]) -> bool:
        """Whether "always allow" may be offered: never for a dangerous call."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        return not Permissions.dangerous(tool_name, target)
