"""The judge: a model that says whether a bash command without a rule may
run without asking. It only removes questions; the rules decide first."""

from __future__ import annotations

import logging
import os
import typing as t

import httpx

from ...config import setting

logger = logging.getLogger(__name__)


class Judge:
    """Asks a decisions model on OpenRouter how sure it is that a command is
    safe. Any failure is "not sure": the user is asked, as without a judge."""

    URL: t.ClassVar[str] = "https://openrouter.ai/api/alpha/decisions"
    API_KEY_ENV: t.ClassVar[str] = "OPENROUTER_API_KEY"  # read on the host at each call
    MAX_CHARS: t.ClassVar[int] = 4_000  # a longer command is not judged: nothing is cut
    SAFE: t.ClassVar[str] = (
        "The command only reads, searches, lists, builds, installs project dependencies "
        "or runs tests, or changes files inside the project folder in a way that is easy "
        "to redo (creating files or folders, removing build output and caches). Extra "
        "harmless steps the user did not ask for are fine. Dangerous words that are only "
        "text being searched or printed are fine."
    )
    ASK: t.ClassVar[str] = (
        "The command deletes or overwrites the user's own work or anything outside the "
        "project folder, sends data to a server, publishes or rewrites git history, "
        "changes permissions, settings, users, scheduled jobs or system files, kills "
        "processes, runs downloaded, decoded or hidden code, or does something different "
        "from what the model says it does."
    )
    STATE: t.ClassVar[str] = (
        "What the user asked:\n{user}\n\n"
        "What the model says the command does (its claim):\n{description}\n\n"
        "The command:\n{command}"
    )
    NOT_SURE: t.ClassVar[str] = "judge: not sure ({safe:.2f})"
    PASSED: t.ClassVar[str] = "judge: safe ({safe:.2f})"
    NO_ANSWER: t.ClassVar[str] = "judge: no answer"

    def __init__(
        self,
        model: str,
        *,
        min_safe: float = 0.9,  # at or above this the command runs without asking
        seconds: float = 3,  # most the judge may take
        transport: httpx.AsyncBaseTransport | None = None,  # for tests
    ) -> None:
        self.model = model
        self.min_safe = min_safe
        self.seconds = seconds
        self.transport = transport

    @classmethod
    def configured(cls) -> Judge | None:
        """The judge the settings ask for; None when no model is set."""
        if not setting.harness_judge_model:
            return None
        return cls(
            setting.harness_judge_model,
            min_safe=setting.harness_judge_min_safe,
            seconds=setting.harness_judge_seconds,
        )

    async def safe(self, user: str, description: str, command: str) -> float | None:
        """How sure the judge is that ``command`` is safe, 0 to 1; None when
        it could not say."""
        # 1. Without a key, or with a command too long to send whole: no verdict.
        key = os.environ.get(self.API_KEY_ENV)
        if not key or not command.strip() or len(command) > self.MAX_CHARS:
            return None
        # 2. One question with two answers; the description goes as a claim.
        state = self.STATE.format(
            user=user[: self.MAX_CHARS] or "(nothing)",
            description=description[: self.MAX_CHARS] or "(nothing)",
            command=command,
        )
        question = {"type": "choice", "criteria": {"safe": self.SAFE, "ask": self.ASK}}
        body = {"model": self.model, "state": state, "questions": {"verdict": question}}
        # 3. Ask; whatever goes wrong is "no verdict", never an error for the turn.
        try:
            async with httpx.AsyncClient(timeout=self.seconds, transport=self.transport) as client:
                response = await client.post(
                    self.URL, headers={"Authorization": f"Bearer {key}"}, json=body
                )
                response.raise_for_status()
                return self.read(response.json())
        except Exception as error:
            logger.warning("The judge gave no verdict: %s", type(error).__name__)
            return None

    def passes(self, safe: float | None) -> bool:
        """True when the verdict is sure enough to run without asking."""
        return safe is not None and safe >= self.min_safe

    def note(self, safe: float | None) -> str:
        """The verdict in one line, for the approval card and the trace."""
        if safe is None:
            return self.NO_ANSWER
        return (self.PASSED if self.passes(safe) else self.NOT_SURE).format(safe=safe)

    @staticmethod
    def read(answer: t.Any) -> float | None:
        """The probability of "safe" in the model's answer; None when the
        answer is not the shape we asked for."""
        try:
            verdict = answer["answers"]["verdict"]
            safe = float(verdict["probabilities"]["safe"])
        except (KeyError, TypeError, ValueError):
            return None
        # The choice and its probability must agree, and stay in range.
        if not 0 <= safe <= 1 or (verdict.get("choice") != "safe" and safe >= 0.5):
            return None
        return safe
