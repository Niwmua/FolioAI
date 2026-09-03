"""Exception hierarchy.

Every user-facing error answers three questions: what failed, why, and what to do next.
That contract is enforced structurally -- ``FolioError`` takes a ``remedy`` argument and
``format_for_user`` renders it -- so a bare ``raise SomeError("broke")`` is visibly missing
its remedy at the call site rather than at 2am in a traceback.

The brief names this hierarchy ``BooktransError``; the package is ``folioai`` (DECISIONS Q1),
so the root is ``FolioError`` with the same children.
"""

from __future__ import annotations

__all__ = [
    "BudgetExceeded",
    "ConfigError",
    "EvaluationError",
    "ExtractionError",
    "FolioError",
    "LLMError",
    "RateLimitError",
    "RenderError",
    "StoreError",
    "ValidationError",
]


class FolioError(Exception):
    """Base class for every error this application raises deliberately.

    Args:
        message: What failed, in one sentence, in the user's terms.
        remedy: What the user should do next. Omit only when there is genuinely
            nothing actionable -- an internal invariant violation, for instance.
        context: Structured detail for logs. Never rendered into the terminal
            message, so it may hold paths, ids and counts freely (but never secrets).
    """

    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        remedy: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.context: dict[str, object] = context or {}

    def format_for_user(self) -> str:
        """Render the error as the terminal should show it."""
        if self.remedy:
            return f"{self.message}\n\nWhat to do: {self.remedy}"
        return self.message

    def __str__(self) -> str:
        return self.message


class ConfigError(FolioError):
    """Configuration is missing, malformed, or internally contradictory."""

    exit_code = 2


class ExtractionError(FolioError):
    """A PDF could not be probed, extracted, or cleaned into usable text."""

    exit_code = 3


class LLMError(FolioError):
    """An LLM call failed in a way retrying will not fix."""

    exit_code = 4


class RateLimitError(LLMError):
    """The endpoint rate-limited us.

    Distinct from ``LLMError`` because it is transient and the retry policy treats it
    specially: honour ``Retry-After``, back off, and count against the transient cap
    rather than the translation attempt ladder.
    """

    exit_code = 5

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        remedy: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, remedy=remedy, context=context)
        self.retry_after = retry_after


# N818: the brief names this exception BudgetExceeded; keeping the agreed vocabulary
# beats satisfying a naming lint.
class BudgetExceeded(FolioError):  # noqa: N818
    """The job hit its ``--max-cost`` ceiling and stopped cleanly."""

    exit_code = 6


class ValidationError(FolioError):
    """Deterministic post-translation validation found a critical defect."""

    exit_code = 7


class EvaluationError(FolioError):
    """The evaluator's response could not be parsed into the expected schema."""

    exit_code = 8


class RenderError(FolioError):
    """An export format could not be produced."""

    exit_code = 9


class StoreError(FolioError):
    """The job database is missing, locked, or at an unexpected schema version."""

    exit_code = 10
