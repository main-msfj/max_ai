"""Declarative command permissions, separate from approval and execution."""

from __future__ import annotations

import shlex
from fnmatch import fnmatchcase
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

BashPermission = Literal["allow", "ask", "deny"]

# Commands that run their arguments as another command. Deny and ask patterns
# are checked against the wrapped command too, so ``env sudo`` is still sudo.
_WRAPPERS = {
    "builtin", "command", "env", "exec", "nice", "nohup", "setsid",
    "stdbuf", "time", "timeout", "xargs",
}
# Wrapper options whose value is the next word (``nice -n 5``, ``env -u VAR``).
_WRAPPER_VALUE_OPTIONS = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "exec": {"-a"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "xargs": {"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s"},
}
_SHELLS = {"bash", "dash", "ksh", "sh", "zsh"}
_MAX_NESTING = 4


class BashPermissions(BaseModel):
    """Patterns match literal command words; a trailing :* permits extra args.

    Each supplied list replaces its defaults; [] disables that category.
    Unknown commands and unsupported shell syntax require approval. This is
    a conservative permission classifier, not a shell parser or a sandbox.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    allowed_patterns: list[StrictStr] = Field(
        default_factory=lambda: ["pwd", "git status"], strict=True,
    )
    ask_patterns: list[StrictStr] = Field(
        default_factory=lambda: ["git push:*"], strict=True,
    )
    deny_patterns: list[StrictStr] = Field(
        default_factory=lambda: [
            "sudo:*", "su:*", "doas:*", "shutdown:*", "reboot:*",
            "halt:*", "poweroff:*", "mkfs:*", "mount:*", "umount:*",
            "rm -rf /*", "git push --force:*",
        ], strict=True,
    )

    @field_validator("allowed_patterns", "ask_patterns", "deny_patterns")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            if not pattern.strip() or any(c in pattern for c in "\0\n\r"):
                raise ValueError("Patterns must be non-empty single-line strings")
            words = shlex.split(pattern.removesuffix(":*"))
            if not words:
                raise ValueError("Patterns must name a command")
        return list(patterns)

    @staticmethod
    def _matches(words: list[str], pattern: str) -> bool:
        prefix = pattern.endswith(":*")
        expected = shlex.split(pattern[:-2] if prefix else pattern)
        if len(words) < len(expected) or (not prefix and len(words) != len(expected)):
            return False
        return all(fnmatchcase(word, glob) for word, glob in zip(words, expected))

    def evaluate(self, command: str, _depth: int = 0) -> BashPermission:
        """Allow a compound command only when every literal segment is allowed.

        Expansions, quoting, redirection, background jobs, subshells and control
        structures fall back to ask even if an allow pattern matches. Visible
        deny matches are still checked first, also through command paths
        (``/usr/bin/sudo``), wrappers (``env sudo``) and inline scripts
        (``bash -c "sudo ..."``, ``eval``). Dynamic code is not interpreted,
        so deny patterns are a guard rail, not a sandbox.
        """
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            return "deny"
        uncertain = any(c in command for c in "$`\\\"'(){}<>*?[]#\r")
        lexer = shlex.shlex(command.replace("\n", ";"), posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError:
            return "ask"
        segments: list[list[str]] = []
        words: list[str] = []
        for token in tokens:
            if token and all(c in ";&|()<>" for c in token):
                if not words or token not in {";", "&&", "||", "|"}:
                    uncertain = True
                if words:
                    segments.append(words)
                    words = []
            else:
                words.append(token)
        if words:
            segments.append(words)
        else:
            uncertain = True
        if not segments:
            return "ask"
        decisions: list[BashPermission] = []
        for words in segments:
            candidates = _unwrap(words)
            if any(
                self._matches(candidate, pattern)
                for candidate in candidates for pattern in self.deny_patterns
            ) or any(
                self.evaluate(inner, _depth + 1) == "deny"
                for inner in _inline_scripts(candidates)
                if _depth < _MAX_NESTING
            ):
                return "deny"
            if len(candidates) > 1:
                # Wrapped or path-qualified commands always need approval.
                uncertain = True
            if any(
                self._matches(candidate, pattern)
                for candidate in candidates for pattern in self.ask_patterns
            ):
                decisions.append("ask")
            elif any(self._matches(words, pattern) for pattern in self.allowed_patterns):
                decisions.append("allow")
            else:
                decisions.append("ask")
            # Shell keywords/assignments are not ordinary executable commands.
            if "=" in words[0] or words[0] in {
                "if", "then", "else", "elif", "fi", "for", "while", "until",
                "do", "done", "case", "esac", "function", "!", "time", "coproc",
            }:
                uncertain = True
        return "ask" if uncertain or "ask" in decisions else "allow"


def _unwrap(words: list[str]) -> list[list[str]]:
    """Return ``words`` plus the command it actually runs, if that differs.

    Strips leading ``VAR=value`` assignments and wrapper commands, and
    reduces the command word to its basename.
    """
    candidates = [words]
    rest = list(words)
    for _ in range(len(words)):
        while rest and "=" in rest[0] and not rest[0].startswith("="):
            rest = rest[1:]
        if not rest:
            break
        name = rest[0].rsplit("/", 1)[-1]
        rest = [name, *rest[1:]]
        if name not in _WRAPPERS:
            break
        value_options = _WRAPPER_VALUE_OPTIONS.get(name, set())
        rest = rest[1:]
        while rest and rest[0].startswith("-") and rest[0] != "-":
            option = rest.pop(0)
            if option == "--":
                break
            if option in value_options and rest:
                rest.pop(0)
        if name == "timeout" and rest:
            rest = rest[1:]  # the duration
    if rest and rest != words:
        candidates.append(rest)
    return candidates


def _inline_scripts(candidates: list[list[str]]) -> list[str]:
    """Scripts passed as strings: ``sh -c '...'``, ``bash -lc '...'``, ``eval``."""
    scripts: list[str] = []
    for words in candidates:
        name = words[0].rsplit("/", 1)[-1]
        if name == "eval" and len(words) > 1:
            scripts.append(" ".join(words[1:]))
        elif name in _SHELLS:
            for index, word in enumerate(words[1:], start=1):
                if not word.startswith("-"):
                    break
                if "c" in word.lstrip("-") and not word.startswith("--"):
                    if index + 1 < len(words):
                        scripts.append(words[index + 1])
                    break
    return scripts
