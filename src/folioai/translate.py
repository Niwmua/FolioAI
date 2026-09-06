"""Translation engine (brief §8).

Builds the prompt for a batch, sends it, parses the tagged response, and reports exactly
what came back. It does not decide whether a translation is *good* -- that is
``validate.py`` and ``evaluate.py`` -- and it does not decide what to do about a bad one,
which is ``orchestrate.py``. Keeping those apart is what makes the retry ladder testable
without a model.

The rolling summary lives here too, because it is a translation-time concern: it is the
thing that stops a 400-page novel drifting in register halfway through (§6).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .glossary import Glossary
from .llm.client import LLMClient, LLMResponse, Message
from .logging_setup import get_logger
from .prompts import SUMMARIZE_SYSTEM, TRANSLATE_RETRY, TRANSLATE_SYSTEM, render
from .segment import Batch, BatchContext
from .tags import ParsedSegments, parse_segments, tag_overhead

if TYPE_CHECKING:
    from .config import Settings
    from .ir import Document

log = get_logger(__name__)

#: Style-profile keys the translator prompt reads directly rather than listing generically.
_DIRECT_KEYS = {"dialogue_convention", "idioms", "measurements", "notes", "name"}


@dataclass(slots=True)
class RetryContext:
    """What attempt 2 and 3 add to the prompt (Appendix B)."""

    previous_output: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    mechanical_problems: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.issues or self.mechanical_problems)


@dataclass(slots=True)
class BatchTranslation:
    """One attempt at one batch, described rather than judged."""

    batch: Batch
    attempt_no: int
    model: str
    response: LLMResponse
    parsed: ParsedSegments
    messages: list[Message]

    @property
    def texts(self) -> dict[str, str]:
        return self.parsed.texts

    @property
    def missing(self) -> list[str]:
        return self.parsed.missing(self.batch.ids)

    @property
    def unexpected(self) -> list[str]:
        return self.parsed.unexpected(self.batch.ids)

    @property
    def blocked(self) -> dict[str, str]:
        return self.parsed.blocked

    @property
    def complete(self) -> bool:
        """Every requested id came back exactly once, in order, with nothing invented."""
        return (
            not self.missing
            and not self.unexpected
            and not self.parsed.duplicates
            and not self.parsed.out_of_order(self.batch.ids)
        )


class Translator:
    """Turns batches into translated segments."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings,
        *,
        document: Document,
        target_lang: str,
        style_profile: dict[str, Any] | None = None,
        glossary: Glossary | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.document = document
        self.target_lang = target_lang
        self.style_profile = style_profile or {}
        self.glossary = glossary or Glossary()

    # -- prompt construction --------------------------------------------------------

    def build_system_prompt(self, batch: Batch, context: BatchContext) -> str:
        """Render the translator system prompt for one batch (Appendix A)."""
        source_text = "\n".join(unit.text for unit in batch.units)
        terms = self.glossary.for_text(source_text)
        profile = self.style_profile

        return render(
            TRANSLATE_SYSTEM,
            source_lang=self.document.source_lang,
            target_lang=self.target_lang,
            title=self.document.title,
            author=self.document.author,
            dialogue_convention=profile.get(
                "dialogue_convention", f"the standard conventions of {self.target_lang}."
            ),
            idiom_policy=profile.get("idioms", "equivalent"),
            measurement_policy=profile.get("measurements", "preserve"),
            profile_notes=profile.get("notes") or [],
            glossary=[term.model_dump() for term in terms],
            style_profile={
                key: value
                for key, value in profile.items()
                if key not in _DIRECT_KEYS and not isinstance(value, list | dict)
            },
            rolling_summary=context.rolling_summary,
            previous_target=context.previous_target,
            next_source=context.next_source,
        )

    def build_messages(
        self,
        batch: Batch,
        context: BatchContext,
        *,
        retry: RetryContext | None = None,
    ) -> list[Message]:
        """System prompt, the tagged batch, and -- on a retry -- the correction request."""
        messages: list[Message] = [
            {"role": "system", "content": self.build_system_prompt(batch, context)},
            {"role": "user", "content": batch.render()},
        ]
        if retry is not None:
            messages.append(
                {
                    "role": "user",
                    "content": render(
                        TRANSLATE_RETRY,
                        previous_output=retry.previous_output,
                        issues=retry.issues,
                        mechanical_problems=retry.mechanical_problems,
                    ),
                }
            )
        return messages

    # -- the call ---------------------------------------------------------------------

    def completion_budget(self, batch: Batch, *, widen: int = 0) -> int:
        """The ``max_tokens`` for one batch.

        Four parts, and only the first is about how long the book is:

        * **the translation itself** -- source tokens times ``max_completion_ratio``, which
          brackets how far a language pair can expand;
        * **the tag protocol** -- the model has to echo a ``<seg>`` wrapper for every block,
          and that cost scales with segment *count*, not with prose. Leaving it out
          under-budgets an ordinary batch by ~800 tokens and a table of contents by 2,600;
        * **reasoning headroom** -- a flat allowance for models that think before they
          write. Those tokens are charged against ``max_tokens`` and never appear in the
          response, so a budget sized for the visible answer alone is spent on thinking and
          the endpoint stops for length partway through the first segment;
        * **the widening** -- each truncated attempt doubles the budget for the next one,
          because retrying a length failure with the same ceiling produces the same failure.

        ``widen`` is bounded by ``retry.max_attempts``, so the doubling cannot run away.
        """
        cfg = self.settings.translation
        budget = int(batch.source_tokens * cfg.max_completion_ratio) + 512
        budget += tag_overhead(batch.size)
        budget += cfg.reasoning_headroom_tokens
        # Annotated because int.__pow__ is typed as returning Any (negative exponents give
        # a float), which would leak Any into the return type under strict mode.
        doubling: int = 2**widen
        return budget * doubling

    async def translate_batch(
        self,
        batch: Batch,
        context: BatchContext,
        *,
        attempt_no: int = 1,
        model: str | None = None,
        temperature: float | None = None,
        retry: RetryContext | None = None,
        widen: int = 0,
    ) -> BatchTranslation:
        """Translate one batch once. Never retries -- that is the orchestrator's job."""
        model = model or self.settings.models.translator
        if temperature is None:
            temperatures = self.settings.retry.attempt_temperatures
            temperature = temperatures[min(attempt_no - 1, len(temperatures) - 1)]

        messages = self.build_messages(batch, context, retry=retry)
        max_tokens = self.completion_budget(batch, widen=widen)

        response = await self.client.complete(
            messages,
            model=model,
            temperature=temperature,
            top_p=self.settings.translation.top_p,
            max_tokens=max_tokens,
            purpose="translate",
        )
        parsed = parse_segments(response.text)

        result = BatchTranslation(
            batch=batch,
            attempt_no=attempt_no,
            model=model,
            response=response,
            parsed=parsed,
            messages=messages,
        )
        log.info(
            "batch_translated",
            batch=batch.index,
            chapter=batch.chapter_id,
            attempt=attempt_no,
            model=model,
            segments=batch.size,
            returned=len(parsed.order),
            missing=len(result.missing),
            unexpected=len(result.unexpected),
            cached=response.cached,
            latency_ms=response.latency_ms,
            max_tokens=max_tokens,
            truncated=response.truncated,
            reasoning_tokens=response.reasoning_tokens,
        )
        return result


class SummaryTracker:
    """Per-chapter rolling summary (§6).

    Refreshed every ``context.summary_every`` batches with a cheap model. This is the piece
    that keeps a long book from drifting: without it, chapter 20 is translated by a model
    that has no idea who is in the room.
    """

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.summaries: dict[str, str] = {}
        self._since_refresh: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def get(self, chapter_id: str | None) -> str:
        return self.summaries.get(chapter_id or "", "")

    def note_batch(self, chapter_id: str | None) -> bool:
        """Record that a batch was translated; return True when a refresh is due."""
        key = chapter_id or ""
        count = self._since_refresh.get(key, 0) + 1
        self._since_refresh[key] = count
        return count >= self.settings.context.summary_every

    async def refresh(self, chapter_id: str | None, recent_source: list[str]) -> str:
        """Update a chapter's summary from the text just translated."""
        key = chapter_id or ""
        if not recent_source:
            return self.summaries.get(key, "")

        async with self._lock:
            system = render(
                SUMMARIZE_SYSTEM,
                max_words=self.settings.context.summary_max_words,
                current_summary=self.summaries.get(key, ""),
            )
            response = await self.client.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(recent_source)},
                ],
                model=self.settings.models.role("summarizer"),
                temperature=0.0,
                max_tokens=self.settings.context.summary_max_words * 3,
                purpose="summarize",
            )
            summary = response.text.strip()
            self.summaries[key] = summary
            self._since_refresh[key] = 0

        log.info("summary_refreshed", chapter=chapter_id, words=len(summary.split()))
        return summary
