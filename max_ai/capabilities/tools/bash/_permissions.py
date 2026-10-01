"""How the Policy reads a bash command: its segments, and whether a rule
pattern matches one. Rules themselves live in ``core.policy.Policy``."""

from __future__ import annotations

import shlex
from fnmatch import fnmatchcase

_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "until",
    "do", "done", "case", "esac", "function", "!", "time", "coproc",
}


def split_command(command: str) -> tuple[list[str], bool]:
    """The simple commands in ``command`` (``ls && rm x`` → ``["ls", "rm x"]``)
    and whether they are all of it.

    Expansions, quoting, redirection, background jobs, subshells and control
    structures make it uncertain: deny and ask rules still see the visible
    segments, but nothing is allowed without asking.
    """
    if not isinstance(command, str) or not command.strip() or "\0" in command:
        return [], False
    certain = not any(c in command for c in "$`\\\"'(){}<>*?[]#\r")
    lexer = shlex.shlex(command.replace("\n", ";"), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return [command.strip()], False
    segments: list[list[str]] = []
    words: list[str] = []
    for token in tokens:
        if token and all(c in ";&|()<>" for c in token):
            if not words or token not in {";", "&&", "||", "|"}:
                certain = False
            if words:
                segments.append(words)
                words = []
        else:
            words.append(token)
    if words:
        segments.append(words)
    else:
        certain = False
    # Shell keywords and assignments are not ordinary commands.
    if any("=" in words[0] or words[0] in _KEYWORDS for words in segments):
        certain = False
    return [shlex.join(words) for words in segments], certain and bool(segments)


def command_matches(command: str, pattern: str) -> bool:
    """``command`` (one segment) matches ``pattern`` word by word; a trailing
    ``:*`` allows more arguments (``git push:*``)."""
    try:
        words = shlex.split(command)
        prefix = pattern.endswith(":*")
        expected = shlex.split(pattern[:-2] if prefix else pattern)
    except ValueError:
        return False
    if not expected or len(words) < len(expected) or (not prefix and len(words) != len(expected)):
        return False
    return all(fnmatchcase(word, glob) for word, glob in zip(words, expected))
