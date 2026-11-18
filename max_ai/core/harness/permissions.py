"""Who decides if a tool call runs: the user's rules in ``setting.json`` and
the commands the harness never lets run without asking."""

import fnmatch
import re
import shlex
import typing as t
from collections.abc import Mapping

from ...base.layout import Layout


class Permissions:
    """Rules are ``tool`` or ``tool(pattern)``; the pattern is matched with
    the bash command or the path of a file tool. ``decide`` answers False
    (denied), None (ask the user), True (run) or ``DEFAULT`` (no rule)."""

    DEFAULT: t.ClassVar[str] = "default"
    KINDS: t.ClassVar[tuple[str, ...]] = ("allow", "ask", "deny")
    BASH: t.ClassVar[str] = "bash"
    SETTINGS: t.ClassVar[str] = Layout.SETTINGS
    # The skills folder as a command names it: under .agent, or reached with "..".
    SKILLS: t.ClassVar[str] = rf"(^|/)({re.escape(Layout.FOLDER)}|\.\.)/{Layout.SKILLS}(/|$)"
    # Commands that change the files they are given.
    WRITERS: t.ClassVar[frozenset[str]] = frozenset({
        "cp", "mv", "tee", "touch", "rm", "rmdir", "mkdir", "chmod", "chown", "ln",
        "truncate", "dd", "install", "sed",
    })
    GLOB: t.ClassVar[str] = "*?["  # a word with one of these is a pattern, not a name
    # Always asked, whatever the tool or the user's allow list say.
    DANGEROUS: t.ClassVar[tuple[str, ...]] = (
        r"^rm\b", r"^sudo\b", r"^dd\b", r"^mkfs", r"^shutdown\b", r"^reboot\b",
        r"^(chmod|chown)\s+(-\w*R|--recursive)", r"^git\s+push\b.*(--force|-f\b)",
        r"^git\s+reset\s+--hard", r"^git\s+clean\b", r"^kill(all)?\b", r"^eval\b",
        r"^(sh|bash|zsh|python3?)$",  # something is piped into an interpreter
    )
    # What cannot be read safely: a command built while it runs.
    UNREADABLE: t.ClassVar[tuple[str, ...]] = ("$(", "`", "<(")
    # Separators split commands only outside quotes: ``python3 -c "a\nb"`` is one.
    PIECES: t.ClassVar[str] = r"""(?P<text>'[^']*'|"(?:\\.|[^"\\])*"|\\.)|(?P<sep>&&|\|\||;|\||\n)"""
    # The text a heredoc feeds a command is data, not commands.
    HEREDOC: t.ClassVar[str] = r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1[ \t]*(?=\n|$)"
    HARMLESS: t.ClassVar[frozenset[str]] = frozenset({"cd"})  # never asked on its own

    def __init__(self, rules: Mapping[str, list[str]] | None = None) -> None:
        given = rules or {}
        self.allow = list(given.get("allow", []))
        self.ask = list(given.get("ask", []))
        self.deny = list(given.get("deny", []))

    def decide(self, tool_name: str, parameters: Mapping[str, t.Any]) -> bool | str | None:
        """What happens with this call; the strictest rule that matches wins.
        A command is judged part by part: ``cd x && pytest`` needs pytest allowed."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        parts = self.parts(target) if tool_name == self.BASH else [target]
        # 1. The user's deny list, for the whole call or any part of it.
        if any(self.matches(self.deny, tool_name, part) for part in [target, *parts]):
            return False
        # 2. What the harness always asks, and the user's ask list.
        if self.dangerous(tool_name, target):
            return None
        if any(self.matches(self.ask, tool_name, part) for part in [target, *parts]):
            return None
        # 3. Every part allowed (a lone cd always is); else the tool's own way.
        if all(self.harmless(tool_name, part) or self.matches(self.allow, tool_name, part)
               for part in parts or [target]):
            return True
        return self.DEFAULT

    @staticmethod
    def parts(command: str) -> list[str]:
        """The commands joined in ``command``, without the text of its heredocs."""
        code = re.sub(Permissions.HEREDOC, "", command, flags=re.S)
        parts: list[str] = []
        start = 0
        for found in re.finditer(Permissions.PIECES, code, flags=re.S):
            if found.group("sep"):
                parts.append(code[start:found.start()])
                start = found.end()
        parts.append(code[start:])
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def harmless(tool_name: str, part: str) -> bool:
        """A command that is fine on its own, like changing folder."""
        words = part.split()
        return tool_name == Permissions.BASH and bool(words) and words[0] in Permissions.HARMLESS

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
        if Permissions.protected(target):
            return True
        # Each part of the command is looked at alone: ``ls && rm x`` has an rm.
        for part in Permissions.parts(target):
            if any(re.search(pattern, part) for pattern in Permissions.DANGEROUS):
                return True
        return False

    @staticmethod
    def protected(command: str) -> bool:
        """A command that reaches the settings, or writes in the skills: the
        files the harness decides from. Each word is read as bash would,
        without its quotes; a pattern counts when it can reach the file."""
        for part in Permissions.parts(command):
            # 1. Quotes that do not close: it cannot be read, so it is asked.
            try:
                raw = shlex.split(part)
            except ValueError:
                return True
            # 2. A redirection glued to its file (``>file``) is the file's word.
            words = [word.lstrip("0123456789<>&") for word in raw]
            into = [
                words[index] if words[index] else "".join(words[index + 1:index + 2])
                for index, word in enumerate(raw)
                if re.match(r"\d*>", word)
            ]
            words = [word for word in words if word]
            if not words:
                continue
            # 3. The settings, by name or by a pattern with a folder before it.
            for word in words:
                folder, _, name = word.rpartition("/")
                pattern = any(mark in name for mark in Permissions.GLOB)
                if fnmatch.fnmatchcase(Permissions.SETTINGS, name) and (folder or not pattern):
                    return True
            # 4. The skills are read and run by anyone, and written by the harness
            #    only: a command that changes files, or output sent into one.
            if any(re.search(Permissions.SKILLS, word) for word in into):
                return True
            name = words[0].rpartition("/")[2]
            in_place = any(word.startswith(("-i", "--in-place")) for word in words)
            writes = name in Permissions.WRITERS and (name != "sed" or in_place)
            if writes and any(re.search(Permissions.SKILLS, word) for word in words):
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
    def rules_for(tool_name: str, parameters: Mapping[str, t.Any]) -> list[str]:
        """The rules "always allow" keeps: one per command of a bash call, by
        its first word (a lone cd needs none); any other tool whole."""
        if tool_name != Permissions.BASH:
            return [tool_name]
        rules: list[str] = []
        for part in Permissions.parts(str(parameters.get("command") or "")):
            rule = f"{tool_name}({part.split()[0]}*)"
            if not Permissions.harmless(tool_name, part) and rule not in rules:
                rules.append(rule)
        return rules

    @staticmethod
    def can_be_kept(tool_name: str, parameters: Mapping[str, t.Any]) -> bool:
        """Whether "always allow" may be offered: never for a dangerous call,
        nor when there is no rule to keep."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        rules = Permissions.rules_for(tool_name, parameters)
        return bool(rules) and not Permissions.dangerous(tool_name, target)
