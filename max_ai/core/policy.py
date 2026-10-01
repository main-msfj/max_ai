"""Whether a tool call runs: allow, ask or deny, decided in one place.

Rules look like Claude Code permissions: ``Tool`` or ``Tool(pattern)``. The
tool name may be a glob (``acme_hr_*``) and is not case sensitive. What a
pattern matches depends on the tool: bash matches each command in the line
(``Bash(git push:*)``), the file tools match the path (``WriteFile(secrets/**)``),
any tool can name a parameter with ``policy_subject``.

deny wins over ask, ask over allow; with no rule the tool's own default
(``approval_mode``) applies. ``Policy()`` is the default every agent gets.
"""

from __future__ import annotations

import fnmatch
import re
import typing as t

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..types.tools import ToolApprovalMode

if t.TYPE_CHECKING:
    from ..base.tools import CoreTool

Verdict = t.Literal["allow", "ask", "deny"]

DEFAULT_ALLOW = ["Bash(pwd)", "Bash(git status)"]
DEFAULT_ASK = ["Bash(git push:*)"]
DEFAULT_DENY = [
    "Bash(sudo:*)", "Bash(su:*)", "Bash(doas:*)", "Bash(shutdown:*)", "Bash(reboot:*)",
    "Bash(halt:*)", "Bash(poweroff:*)", "Bash(mkfs:*)", "Bash(mount:*)", "Bash(umount:*)",
    "Bash(rm -rf /*)", "Bash(git push --force:*)",
]
ON_HOST = "commands on this machine always ask"

_RULE = re.compile(r"^([^()\s]+)(?:\((.+)\))?$", re.DOTALL)


class Decision(t.NamedTuple):
    """The verdict and the rule behind it (None: the tool's default)."""

    verdict: Verdict
    rule: str | None


class Policy(BaseModel):
    """allow / ask / deny rules for every tool of an agent."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOW),
        description="Rules that run without asking, e.g. Bash(npm test:*) or ReadFile.",
    )
    ask: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ASK),
        description="Rules that always ask the user first, e.g. acme_hr_*.",
    )
    deny: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DENY),
        description="Rules that never run, e.g. Bash(sudo:*) or WriteFile(secrets/**).",
    )

    @field_validator("allow", "ask", "deny")
    @classmethod
    def _valid_rules(cls, rules: list[str]) -> list[str]:
        for rule in rules:
            if not isinstance(rule, str) or not _RULE.match(rule.strip()):
                raise ValueError(f"Not a rule: {rule!r}. Use Tool or Tool(pattern).")
        return [rule.strip() for rule in rules]

    def decide(
        self,
        tool: CoreTool,
        parameters: dict[str, t.Any],
        *,
        isolated: bool,
        extra_allow: t.Sequence[str] = (),
    ) -> Decision:
        """allow, ask or deny for this call. ``isolated`` says the executor is a
        sandbox; ``extra_allow`` are rules the user added during the session."""
        subjects, certain = tool.permission_subjects(parameters)
        # deny and ask: one matching subject is enough ("ls && sudo x" is denied).
        for verdict, rules in (("deny", self.deny), ("ask", self.ask)):
            for rule in rules:
                if any(_match(rule, tool, subject) for subject in subjects or [None]):
                    return Decision(t.cast(Verdict, verdict), rule)
        # allow: every subject covered, and the tool sure it saw all of them.
        # The agent's rules on the host still ask; the user's own don't.
        for rules in (self.allow, [*self.allow, *extra_allow]):
            used = _covering(rules, tool, subjects, certain)
            if used:
                decision = Decision("allow", used[0])
                by_user = any(rule in extra_allow for rule in used)
                return decision if by_user else _on_host(tool, decision, isolated)
        default: Verdict = "allow" if _auto(tool) else "ask"
        return _on_host(tool, Decision(default, None), isolated)

    def rules_for(self, tool: CoreTool, parameters: dict[str, t.Any]) -> list[str]:
        """What "always allow" adds for this call: ``Bash(npm:*)`` for each
        command in the line, the bare tool name otherwise; [] when the call
        can't be read well enough to allow it again safely."""
        subjects, certain = tool.permission_subjects(parameters)
        if not certain:
            return []
        if not tool.runs_commands:
            return [tool.name]
        return sorted({f"{tool.name}({subject.split()[0]}:*)" for subject in subjects if subject})


def _covering(
    rules: t.Sequence[str], tool: CoreTool, subjects: list[str], certain: bool
) -> list[str]:
    """The rules that allow every subject, one per subject; [] if any is
    left out. A whole-tool rule (``Bash``) covers even what the tool
    couldn't read."""
    found = [next((r for r in rules if _match(r, tool, s)), None) for s in subjects or [None]]
    if not all(found):
        return []
    used = [t.cast(str, rule) for rule in found]
    return used if certain or all("(" not in rule for rule in used) else []


def _auto(tool: CoreTool) -> bool:
    try:
        return ToolApprovalMode(tool.approval_mode) == ToolApprovalMode.AUTO_APPROVED
    except ValueError:
        return False


def _on_host(tool: CoreTool, decision: Decision, isolated: bool) -> Decision:
    """Without a sandbox, commands run on the user's machine: always ask."""
    if decision.verdict == "allow" and tool.runs_commands and not isolated:
        return Decision("ask", ON_HOST)
    return decision


def _match(rule: str, tool: CoreTool, subject: str | None) -> bool:
    """``rule`` names this tool and, with a pattern, matches ``subject``."""
    found = _RULE.match(rule)
    if found is None:
        return False
    name, pattern = found.groups()
    if not fnmatch.fnmatchcase(tool.name.lower(), name.lower()):
        return False
    return pattern is None or (subject is not None and tool.matches(pattern, subject))


__all__ = ["DEFAULT_ALLOW", "DEFAULT_ASK", "DEFAULT_DENY", "Decision", "Policy", "Verdict"]
