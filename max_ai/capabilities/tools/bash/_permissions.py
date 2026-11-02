"""How the Policy reads a bash command: its segments, and whether a rule
pattern matches one. Rules themselves live in ``core.policy.Policy``."""

from __future__ import annotations

import shlex
from fnmatch import fnmatchcase

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


_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "until",
    "do", "done", "case", "esac", "function", "!", "time", "coproc",
}


def split_command(command: str, _depth: int = 0) -> tuple[list[str], bool]:
    """The simple commands in ``command`` (``ls && rm x`` → ``["ls", "rm x"]``)
    and whether they are all of it.

    Each command also yields what it really runs: ``/usr/bin/sudo x``,
    ``env sudo x`` and ``FOO=1 sudo x`` add ``sudo x``; ``bash -c '...'`` and
    ``eval ...`` add the commands inside. Those, expansions, quoting,
    redirection, background jobs, subshells and control structures make it
    uncertain: deny and ask rules still see every command, but nothing is
    allowed without asking.
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
    subjects: list[str] = []
    for words in segments:
        # Shell keywords and assignments are not ordinary commands.
        if "=" in words[0] or words[0] in _KEYWORDS:
            certain = False
        candidates = _unwrap(words)
        scripts = _inline_scripts(candidates)
        if len(candidates) > 1 or scripts:
            certain = False
        subjects.extend(shlex.join(candidate) for candidate in candidates)
        for script in scripts if _depth < _MAX_NESTING else []:
            subjects.extend(split_command(script, _depth + 1)[0])
    return list(dict.fromkeys(subjects)), certain and bool(segments)


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
