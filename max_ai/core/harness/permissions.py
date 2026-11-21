"""Who decides if a tool call runs: the user's rules in ``setting.json`` and
the commands the harness never lets run without asking."""

import fnmatch
import re
import shlex
import typing as t
from collections.abc import Mapping

from ...base.layout import Layout


class Permissions:
    """Rules are ``tool`` or ``tool(pattern)``. For a file tool the pattern
    is matched with the path. For bash an allow rule is the exact command,
    ``bash(pytest -q)``, or a prefix that ends where a word ends,
    ``bash(git status:*)``; ask and deny rules may also use wildcards.
    ``decide`` answers False (denied), None (ask the user), True (run) or
    ``DEFAULT`` (no rule)."""

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
    # Something is given to an interpreter on its input: alone, or told to read it.
    PIPED: t.ClassVar[tuple[str, ...]] = (
        r"^(sh|bash|zsh|python3?)(\s+-)?$", r"^(sh|bash|zsh)\s+-s\b",
    )
    # Always asked, whatever the tool or the user's allow list say.
    DANGEROUS: t.ClassVar[tuple[str, ...]] = (
        r"^rm\b", r"^sudo\b", r"^dd\b", r"^mkfs", r"^shutdown\b", r"^reboot\b",
        r"^(chmod|chown)\s+(-\w*R|--recursive)", r"^git\s+push\b.*(--force|-f\b)",
        r"^git\s+reset\s+--hard", r"^git\s+clean\b", r"^kill(all)?\b", r"^eval\b", r"^doas\b",
        r"^find\b.*\s-(delete|exec|execdir|ok|okdir)\b",  # find that deletes or runs a command
        *PIPED,
    )
    # Files that hold keys: a call that names one is always asked, whatever the tool.
    SECRETS: t.ClassVar[tuple[str, ...]] = (
        r"(^|[\s/'\"=<>])\.env(\.(?!example|sample|template)[\w-]+)?($|[\s'\";|&)<>])",
        r"\.(pem|key)($|[\s'\";|&)<>])",
        r"\bid_(rsa|ed25519|ecdsa)\b",
        r"\bcredentials(\.\w+)?($|[\s'\";|&)<>])",
        r"(^|[\s/'\"=<>])\.netrc\b",
    )
    # Why a call is asked, said to the user in the approval.
    WHY_SETTINGS: t.ClassVar[str] = "it reaches the settings or the skills: always asked"
    WHY_SECRETS: t.ClassVar[str] = "it names a file of keys: always asked"
    WHY_UNREADABLE: t.ClassVar[str] = "the command cannot be read safely: always asked"
    WHY_DANGEROUS: t.ClassVar[str] = "{command} deletes or takes control: always asked"
    WHY_PIPED: t.ClassVar[str] = "{command} runs code given on its input: always asked"
    WHY_RULE: t.ClassVar[str] = "one of your rules asks for it"
    # Commands that only run another one: what they run is what counts.
    WRAPPERS: t.ClassVar[frozenset[str]] = frozenset({
        "timeout", "env", "nohup", "nice", "command", "time", "xargs", "exec", "stdbuf", "ionice",
    })
    SHELLS: t.ClassVar[frozenset[str]] = frozenset({"sh", "bash", "zsh", "dash"})
    PREFIX: t.ClassVar[str] = ":*"  # how a rule that allows a prefix ends
    # Commands that only read: "always allow" keeps them as a prefix.
    READERS: t.ClassVar[tuple[str, ...]] = (
        "ls", "pwd", "wc", "cat", "head", "tail", "grep",
        "git status", "git diff", "git log", "git show",
    )
    # They run code the model writes: approved once, never kept.
    NEVER_KEPT: t.ClassVar[frozenset[str]] = WRAPPERS | SHELLS | frozenset({
        "python", "python3", "node", "deno", "bun", "perl", "ruby", "php", "eval", "source", ".",
    })
    # Loops and blocks: several commands in one that cannot be told apart.
    BLOCKS: t.ClassVar[frozenset[str]] = frozenset({
        "for", "while", "until", "if", "case", "select", "function", "do", "done", "then", "fi",
    })
    # What cannot be read safely: a command built while it runs.
    UNREADABLE: t.ClassVar[tuple[str, ...]] = ("$(", "`", "<(")
    # Separators split commands only outside quotes: ``python3 -c "a\nb"`` is one.
    PIECES: t.ClassVar[str] = r"""(?P<text>'[^']*'|"(?:\\.|[^"\\])*"|\\.)|(?P<sep>&&|\|\||;|\||\n|(?<![>&])&(?![>&]))"""
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
        if any(self.matches(self.deny, tool_name, part, wide=True) for part in [target, *parts]):
            return False
        # 2. What the harness always asks, and the user's ask list.
        if self.dangerous(tool_name, target):
            return None
        if any(self.matches(self.ask, tool_name, part, wide=True) for part in [target, *parts]):
            return None
        # 3. The whole command as the user approved it; or every part allowed
        #    (a lone cd always is). Else the tool's own way.
        if self.matches(self.allow, tool_name, target):
            return True
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
        """A call the user must see, whatever the rules say."""
        return Permissions.danger(tool_name, target) is not None

    @staticmethod
    def danger(tool_name: str, target: str) -> str | None:
        """Why a call must be seen by the user, in words for them; None when
        the harness does not ask for it: it touches the settings or a file
        of keys, or it is a command that deletes, takes control or cannot be
        read safely."""
        if Permissions.SETTINGS in target:
            return Permissions.WHY_SETTINGS
        # A file of keys, named by any tool.
        if any(re.search(pattern, target) for pattern in Permissions.SECRETS):
            return Permissions.WHY_SECRETS
        if tool_name != Permissions.BASH:
            return None
        if any(mark in target for mark in Permissions.UNREADABLE):
            return Permissions.WHY_UNREADABLE
        if Permissions.protected(target):
            return Permissions.WHY_SETTINGS
        # Each part of the command is looked at alone: ``ls && rm x`` has an rm.
        for part in Permissions.parts(target):
            for command in Permissions.bare(part):
                name = command.split()[0]
                if any(re.search(pattern, command) for pattern in Permissions.PIPED):
                    return Permissions.WHY_PIPED.format(command=name)
                if any(re.search(pattern, command) for pattern in Permissions.DANGEROUS):
                    return Permissions.WHY_DANGEROUS.format(command=name)
        return None

    def why(self, tool_name: str, parameters: Mapping[str, t.Any]) -> str | None:
        """Why the rules ask the user for this call; None when they do not."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        reason = self.danger(tool_name, target)
        if reason is not None:
            return reason
        parts = self.parts(target) if tool_name == self.BASH else [target]
        if any(self.matches(self.ask, tool_name, part, wide=True) for part in [target, *parts]):
            return self.WHY_RULE
        return None

    @staticmethod
    def bare(part: str) -> list[str]:
        """The commands one part of a command really runs, without what
        hides them: brackets, variables set before it, the path of the
        program, a command that only runs another, a shell given its text."""
        # 1. Brackets around it, and the variables set before the program.
        text = part.strip().lstrip("({! \t").rstrip(")} \t")
        try:
            words = shlex.split(text)
        except ValueError:
            return [text]
        while words and re.fullmatch(r"\w+=.*", words[0], flags=re.S):
            words = words[1:]
        if not words:
            return []
        # 2. The program by its name, not its path: /bin/rm is rm.
        words[0] = words[0].rpartition("/")[2]
        found = [" ".join(words)]
        # 3. A shell given its text with -c: the text is a command of its own.
        if words[0] in Permissions.SHELLS and "-c" in words[:-1]:
            inner = words[words.index("-c") + 1]
            found += [command for piece in Permissions.parts(inner) for command in Permissions.bare(piece)]
        # 4. A command that only runs another: every word after it may be that one.
        elif words[0] in Permissions.WRAPPERS:
            for start in range(1, len(words)):
                rest = words[start:]
                rest[0] = rest[0].rpartition("/")[2]
                found.append(" ".join(rest))
        return found

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
    def matches(rules: list[str], tool_name: str, target: str, *, wide: bool = False) -> bool:
        """Whether one of ``rules`` names this tool and, when it has a pattern,
        this target. ``wide``: a bash pattern may use wildcards too (ask and
        deny, where matching more only asks or denies more)."""
        target = target.strip()
        for rule in rules:
            name, _, pattern = rule.partition("(")
            if name.strip() != tool_name:
                continue
            pattern = pattern[:-1] if pattern.endswith(")") else pattern
            if not pattern:
                return True
            # 1. Any other tool: a pattern for its path.
            if tool_name != Permissions.BASH:
                if fnmatch.fnmatchcase(target, pattern):
                    return True
                continue
            # 2. bash: a prefix up to the end of a word, for a command that
            #    sends its output nowhere; or the exact command.
            if pattern.endswith(Permissions.PREFIX):
                prefix = pattern[: -len(Permissions.PREFIX)]
                starts = target == prefix or target.startswith(prefix + " ")
                if starts and not Permissions.redirects(target):
                    return True
            elif target == pattern.strip():
                return True
            if wide and fnmatch.fnmatchcase(target, pattern):
                return True
        return False

    @staticmethod
    def redirects(part: str) -> bool:
        """The command sends its output to a file (``>``, ``--output``): it writes."""
        try:
            words = shlex.split(part)
        except ValueError:
            return True
        return any(re.match(r"\d*&?>", word) or word.startswith("--output") for word in words)

    @staticmethod
    def rules_for(tool_name: str, parameters: Mapping[str, t.Any]) -> list[str]:
        """The rule "always allow" keeps: for bash, a prefix when the command
        only reads, else the exact command (none for a lone cd); any other
        tool whole."""
        if tool_name != Permissions.BASH:
            return [tool_name]
        command = str(parameters.get("command") or "").strip()
        parts = [part for part in Permissions.parts(command) if not Permissions.harmless(tool_name, part)]
        if not parts:
            return []
        # 1. One command that only reads and writes nowhere: its prefix.
        if len(parts) == 1 and not Permissions.redirects(parts[0]):
            for reader in Permissions.READERS:
                if parts[0] == reader or parts[0].startswith(reader + " "):
                    return [f"{tool_name}({reader}{Permissions.PREFIX})"]
        # 2. Anything else: what the user saw, whole.
        return [f"{tool_name}({command})"]

    @staticmethod
    def can_be_kept(tool_name: str, parameters: Mapping[str, t.Any]) -> bool:
        """Whether "always allow" may be offered: never for a dangerous call,
        for what runs code the model writes, for loops and blocks, nor when
        there is no rule to keep."""
        target = str(parameters.get("command") or parameters.get("path") or "")
        if not Permissions.rules_for(tool_name, parameters) or Permissions.dangerous(tool_name, target):
            return False
        if tool_name != Permissions.BASH:
            return True
        for part in Permissions.parts(target):
            # 1. A loop or a block: its commands cannot be told apart.
            if part.startswith(("(", "{")) or part.split()[0] in Permissions.BLOCKS:
                return False
            # 2. An interpreter, or a command that only runs another.
            if any(command.split()[0] in Permissions.NEVER_KEPT for command in Permissions.bare(part)):
                return False
        return True
