"""Declarative command permissions, separate from approval and execution."""

from __future__ import annotations

import shlex
from fnmatch import fnmatchcase
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

BashPermission = Literal["allow", "ask", "deny"]


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
        """Validate patterns for ``BashPermissions``.

Parameters
----------
patterns : list[str]
    Value supplied for ``patterns``."""
        for pattern in patterns:
            if not pattern.strip() or any(c in pattern for c in "\0\n\r"):
                raise ValueError("Patterns must be non-empty single-line strings")
            words = shlex.split(pattern.removesuffix(":*"))
            if not words:
                raise ValueError("Patterns must name a command")
        return list(patterns)

    @staticmethod
    def _matches(words: list[str], pattern: str) -> bool:
        """Perform the internal ``matches`` operation for ``BashPermissions``.

Parameters
----------
words : list[str]
    Value supplied for ``words``.
pattern : str
    Value supplied for ``pattern``."""
        prefix = pattern.endswith(":*")
        expected = shlex.split(pattern[:-2] if prefix else pattern)
        if len(words) < len(expected) or (not prefix and len(words) != len(expected)):
            return False
        return all(fnmatchcase(word, glob) for word, glob in zip(words, expected))

    def evaluate(self, command: str) -> BashPermission:
        """Allow a compound command only when every literal segment is allowed.

        Expansions, quoting, redirection, background jobs, subshells and control
        structures fall back to ask even if an allow pattern matches. Visible
        deny matches are still checked first; dynamic code is not interpreted.
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
            if any(self._matches(words, pattern) for pattern in self.deny_patterns):
                return "deny"
            if any(self._matches(words, pattern) for pattern in self.ask_patterns):
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
