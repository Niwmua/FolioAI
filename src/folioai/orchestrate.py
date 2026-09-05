"""The retry/escalation state machine (brief §11) and the run loop.

Per segment, the ladder is:

| attempt | model               | temp | prompt additions                                  |
|---------|---------------------|------|---------------------------------------------------|
| 1       | ``models.translator``  | 0.2 | glossary + context                              |
| 2       | ``models.translator``  | 0.3 | + previous attempt + evaluator issues           |
| 3       | ``models.escalation``  | 0.0 | + all prior attempts and all prior feedback     |

Every rung of that ladder addresses a model that answered *badly*. A model that stopped for
length never got as far as answering, so a truncated attempt also doubles the next one's
token budget (D-142) -- otherwise the retry reproduces the truncation exactly.

After the final attempt the **highest-scoring attempt is kept**, the segment is marked
``needs_review``, and every attempt stays in the database. Content is never dropped and the
output never has a gap -- a bad translation that is flagged is recoverable; a missing
paragraph nobody noticed is not.

Two safety rails run alongside:

* a **circuit breaker** that stops the run when a whole chapter is failing, because that
  means the prompt, the model or the extraction is broken and grinding through 300 more
  pages of it just burns money (§11, D-34);
* a **budget guard** in the client that stops cleanly at ``--max-cost``.

Everything is written to SQLite as it completes, so SIGINT at any moment leaves a job that
``folioai resume`` finishes exactly once.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import FolioError
from .evaluate import BatchEvaluation, Evaluator, SegmentVerdict, should_evaluate
from .glossary import Glossary
from .llm.client import LLMClient
from .logging_setup import get_logger
from .segment import Batch, BatchContext, Unit, segment_document
from .store import JobStore, SegmentRecord
from .translate import BatchTranslation, RetryContext, SummaryTracker, Translator
from .validate import LengthRatioTracker, ValidationReport, validate_batch

if TYPE_CHECKING:
    from .config import Settings
    from .ir import Document

log = get_logger(__name__)


# N818: 'CircuitBreakerTripped' says what happened; 'CircuitBreakerTrippedError' does
# not say it any better.
class CircuitBreakerTripped(FolioError):  # noqa: N818
    """A whole chapter is failing: stop before spending the rest of the budget on it."""

    exit_code = 11


@dataclass(slots=True)
class Attempt:
    """One attempt at one segment, with whatever verdict it earned."""

    attempt_no: int
    model: str
    text: str
    verdict: SegmentVerdict | None = None
    validation: list[str] = field(default_factory=list)
    #: The check *names* behind ``validation``, whose entries are human-readable prose.
    #: Deciding what to do about a failure means matching on the check, not on its wording.
    checks: list[str] = field(default_factory=list)
    attempt_row_id: int | None = None

    @property
    def score(self) -> float:
        """Score for ranking attempts.

        An attempt that failed deterministic validation ranks below any evaluated one:
        a malformed or truncated response is worse than a merely mediocre translation,
        whatever a judge might have said about it.
        """
        if self.validation:
            return -1.0
        return self.verdict.composite if self.verdict and self.verdict.score else 0.0

    @property
    def passed(self) -> bool:
        return not self.validation and bool(self.verdict and self.verdict.passed)


@dataclass(slots=True)
class SegmentOutcome:
    """Where a segment ended up after the ladder ran."""

    segment_id: str
    text: str
    score: float | None
    needs_review: bool
    attempts: list[Attempt]
    reason: str = ""

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)


@dataclass(slots=True)
class RunStats:
    """Live counters for the progress display and the final summary."""

    total_segments: int = 0
    completed: int = 0
    needs_review: int = 0
    retries: int = 0
    batches_done: int = 0
    batches_total: int = 0
    scores: list[float] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def mean_score(self) -> float:
        return round(sum(self.scores) / len(self.scores), 1) if self.scores else 0.0


class CircuitBreaker:
    """Trips when a chapter fails systematically rather than unluckily (D-34).

    Requires both a failure *rate* over the threshold and a minimum absolute count, so a
    short chapter with two bad segments does not halt a 400-page run.
    """

    def __init__(self, settings: Settings) -> None:
        self.rate = settings.retry.breaker_failure_rate
        self.minimum = settings.retry.breaker_min_failures
        self.attempted: dict[str, int] = {}
        self.failed: dict[str, int] = {}

    def record(self, chapter_id: str | None, *, failed: bool) -> None:
        key = chapter_id or ""
        self.attempted[key] = self.attempted.get(key, 0) + 1
        if failed:
            self.failed[key] = self.failed.get(key, 0) + 1

    def check(self, chapter_id: str | None) -> None:
        key = chapter_id or ""
        failures = self.failed.get(key, 0)
        attempts = self.attempted.get(key, 0)
        if failures < self.minimum or attempts == 0:
            return
        if failures / attempts < self.rate:
            return
        raise CircuitBreakerTripped(
            f"Chapter {key or '(unnamed)'} is failing systematically: {failures} of "
            f"{attempts} segments failed on their first attempt.",
            remedy=(
                "That pattern means the prompt, the model or the extraction is broken, not "
                "that the book is hard. Check 'folioai extract' output for this chapter, "
                "try a different --translator-model, then resume the job."
            ),
            context={"chapter": key, "failed": failures, "attempted": attempts},
        )


class Orchestrator:
    """Runs a whole job: batches out, verdicts in, everything persisted as it goes."""

    def __init__(
        self,
        *,
        client: LLMClient,
        settings: Settings,
        document: Document,
        target_lang: str,
        store: JobStore,
        job_id: str,
        style_profile: dict[str, Any] | None = None,
        glossary: Glossary | None = None,
        on_progress: Callable[[RunStats], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.document = document
        self.store = store
        self.job_id = job_id
        self.glossary = glossary or Glossary()
        self.on_progress = on_progress
        self.rng = rng or random.Random(settings.llm.seed or 0)

        self.translator = Translator(
            client,
            settings,
            document=document,
            target_lang=target_lang,
            style_profile=style_profile,
            glossary=self.glossary,
        )
        self.evaluator = Evaluator(
            client,
            settings,
            source_lang=document.source_lang,
            target_lang=target_lang,
            style_profile=style_profile,
            glossary=self.glossary,
        )
        self.summaries = SummaryTracker(client, settings)
        self.length_tracker = LengthRatioTracker()
        self.breaker = CircuitBreaker(settings)
        self.stats = RunStats()
        self.translations: dict[str, str] = {}
        self.outcomes: dict[str, SegmentOutcome] = {}
        self._units: list[Unit] = []
        self._position: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # -- entry point -------------------------------------------------------------------

    async def run(self, *, only_pending: bool = True) -> dict[str, SegmentOutcome]:
        """Translate the document, skipping anything already finished.

        Args:
            only_pending: Skip segments already marked ``done`` in the store. This is what
                makes ``resume`` do no duplicated work.
        """
        batches = segment_document(self.document, self.settings)
        done = self._already_done() if only_pending else {}
        self.translations.update(done)

        pending = [
            batch
            for batch in (self._strip_done(batch, done) for batch in batches)
            if batch is not None
        ]
        self._index_units(pending)
        self.stats.batches_total = len(pending)
        self.stats.total_segments = sum(batch.size for batch in pending)
        self.stats.completed = 0

        log.info(
            "run_start",
            job=self.job_id,
            batches=len(pending),
            segments=self.stats.total_segments,
            already_done=len(done),
        )
        if not pending:
            return self.outcomes

        semaphore = asyncio.Semaphore(self.settings.translation.concurrency)

        async def worker(batch: Batch) -> None:
            async with semaphore:
                await self._process_batch(batch)

        tasks = [asyncio.create_task(worker(batch)) for batch in pending]
        try:
            for task in asyncio.as_completed(tasks):
                await task
        except BaseException:
            # Whatever went wrong -- budget, breaker, a dead endpoint, Ctrl-C -- stop the
            # rest immediately and wait for them to unwind. Cancelling only the failures we
            # anticipated leaves the others running against a job that is already over, and
            # their writes would land after the summary said the run had stopped.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        log.info(
            "run_complete",
            job=self.job_id,
            completed=self.stats.completed,
            needs_review=self.stats.needs_review,
            mean_score=self.stats.mean_score,
            retries=self.stats.retries,
        )
        return self.outcomes

    # -- one batch ----------------------------------------------------------------------

    async def _process_batch(self, batch: Batch) -> None:
        context = self._context_for(batch)
        self.store.set_segment_status(self.job_id, batch.ids, "translating")

        attempts_by_segment: dict[str, list[Attempt]] = {sid: [] for sid in batch.ids}
        retry_context: RetryContext | None = None
        pending_ids = list(batch.ids)
        max_attempts = self.settings.retry.max_attempts
        # Each attempt that stopped for length doubles the next one's token budget. Without
        # this the ladder retries a truncation against the identical ceiling and gets the
        # identical cut-off response three times over.
        widen = 0

        for attempt_no in range(1, max_attempts + 1):
            model = self._model_for(attempt_no)
            result = await self.translator.translate_batch(
                batch,
                context,
                attempt_no=attempt_no,
                model=model,
                retry=retry_context,
                widen=widen,
            )
            if result.response.truncated:
                widen += 1
            validation = validate_batch(
                result,
                self.settings,
                glossary=self.glossary,
                length_tracker=self.length_tracker,
            )

            evaluation: BatchEvaluation | None = None
            if validation.has_critical:
                # §9: do not pay a judge to tell you the response was malformed.
                log.info(
                    "skipping_evaluation",
                    batch=batch.index,
                    attempt=attempt_no,
                    reason="critical validation findings",
                    findings=[f.check for f in validation.critical],
                )
            elif should_evaluate(batch, validation, self.glossary, self.settings, rng=self.rng):
                evaluation = await self.evaluator.evaluate(result, validation)

            self._record_attempt(batch, result, validation, evaluation, attempts_by_segment)

            if attempt_no == 1:
                for segment_id in batch.ids:
                    failed = not attempts_by_segment[segment_id][-1].passed
                    self.breaker.record(batch.chapter_id, failed=failed)
                self.breaker.check(batch.chapter_id)

            pending_ids = [sid for sid in batch.ids if not attempts_by_segment[sid][-1].passed]
            if not pending_ids:
                break
            if attempt_no == max_attempts:
                break

            self.stats.retries += 1
            retry_context = self._build_retry_context(result, validation, evaluation)
            log.info(
                "retrying_batch",
                batch=batch.index,
                next_attempt=attempt_no + 1,
                failing=len(pending_ids),
                model=self._model_for(attempt_no + 1),
                # Named max_tokens, not token_budget: the log redactor masks any key that
                # is exactly "token" between underscores, and a redacted budget is exactly
                # the number you came to the log to read.
                max_tokens=self.translator.completion_budget(batch, widen=widen),
            )

        await self._finish_batch(batch, attempts_by_segment)

    def _record_attempt(
        self,
        batch: Batch,
        result: BatchTranslation,
        validation: ValidationReport,
        evaluation: BatchEvaluation | None,
        attempts_by_segment: dict[str, list[Attempt]],
    ) -> None:
        """Persist one attempt per segment, with its evaluation if there was one."""
        for segment_id in batch.ids:
            text = result.texts.get(segment_id, "")
            relevant = [
                finding
                for finding in validation.critical
                if finding.segment_id in (segment_id, None)
            ]
            segment_findings = [finding.describe() for finding in relevant]
            segment_checks = [finding.check for finding in relevant]
            verdict = evaluation.verdicts.get(segment_id) if evaluation else None

            row_id = self.store.record_attempt(
                job_id=self.job_id,
                segment_id=segment_id,
                attempt_no=result.attempt_no,
                model=result.model,
                params={
                    **result.response.params,
                    "batch": batch.index,
                    "cached": result.response.cached,
                },
                output_text=text,
                latency_ms=result.response.latency_ms,
                prompt_tokens=result.response.prompt_tokens,
                completion_tokens=result.response.completion_tokens,
                cost_usd=result.response.cost.usd,
            )
            if verdict is not None and evaluation is not None:
                self.store.record_evaluation(
                    attempt_id=row_id,
                    evaluator_model=evaluation.evaluator_model,
                    scores=verdict.score.model_dump() if verdict.score else {},
                    issues=[issue.model_dump() for issue in verdict.issues],
                    composite=verdict.composite,
                    passed=verdict.passed,
                )

            attempts_by_segment[segment_id].append(
                Attempt(
                    attempt_no=result.attempt_no,
                    model=result.model,
                    text=text,
                    verdict=verdict,
                    validation=segment_findings,
                    checks=segment_checks,
                    attempt_row_id=row_id,
                )
            )

        if result.response.cost.usd:
            self.store.record_usage(
                job_id=self.job_id,
                model=result.model,
                endpoint="chat.completions",
                prompt_tokens=result.response.prompt_tokens,
                completion_tokens=result.response.completion_tokens,
                cost_usd=result.response.cost.usd,
            )
            self.stats.cost_usd += result.response.cost.usd

    async def _finish_batch(
        self, batch: Batch, attempts_by_segment: dict[str, list[Attempt]]
    ) -> None:
        """Keep the best attempt for every segment and commit it (§11)."""
        for segment_id in batch.ids:
            attempts = attempts_by_segment[segment_id]
            outcome = self._best_outcome(segment_id, attempts, batch)
            self.outcomes[segment_id] = outcome
            self.translations[segment_id] = outcome.text

            self.store.finalize_segment(
                self.job_id,
                segment_id,
                final_text=outcome.text,
                final_score=outcome.score,
                needs_review=outcome.needs_review,
                status="review" if outcome.needs_review else "done",
            )
            self.stats.completed += 1
            if outcome.needs_review:
                self.stats.needs_review += 1
            if outcome.score is not None:
                self.stats.scores.append(outcome.score)

        self.stats.batches_done += 1
        if self.on_progress:
            self.on_progress(self.stats)

        if self.summaries.note_batch(batch.chapter_id):
            await self.summaries.refresh(batch.chapter_id, [unit.text for unit in batch.units])

    def _best_outcome(
        self, segment_id: str, attempts: list[Attempt], batch: Batch
    ) -> SegmentOutcome:
        """Pick what to keep. Never returns empty text for a segment that has any."""
        passing = [a for a in attempts if a.passed]
        if passing:
            best = passing[-1]  # the last passing attempt is the one with the most feedback
            return SegmentOutcome(
                segment_id=segment_id,
                text=best.text,
                score=best.verdict.composite if best.verdict and best.verdict.score else None,
                needs_review=False,
                attempts=attempts,
            )

        # Nothing passed. Keep the highest-scoring attempt that actually produced text, and
        # flag it -- a flagged bad translation is recoverable, a silent gap is not (§11).
        with_text = [a for a in attempts if a.text.strip()]
        if with_text:
            best = max(with_text, key=lambda a: a.score)
            reason = (
                best.verdict.reason
                if best.verdict and best.verdict.reason
                else "; ".join(best.validation) or "no attempt passed"
            )
        else:
            # Every attempt came back empty. Fall back to the source text so the export is
            # complete and obviously untranslated, rather than silently missing a paragraph.
            source = batch.source_map().get(segment_id, "")
            best = Attempt(attempt_no=0, model="none", text=source)
            truncated_throughout = bool(attempts) and all(
                "truncation" in a.checks for a in attempts
            )
            if truncated_throughout:
                # The distinctive signature of a reasoning model whose thinking ate the
                # whole completion budget: nothing wrong with the prompt or the model, the
                # answer simply never got room to be written.
                reason = (
                    "every attempt stopped for length before this segment was reached; "
                    "source text kept as a placeholder"
                )
                log.error(
                    "segment_never_translated",
                    segment_id=segment_id,
                    batch=batch.index,
                    cause="truncation",
                    remedy=(
                        "the model ran out of completion budget on every attempt; raise "
                        "translation.reasoning_headroom_tokens (FOLIOAI_TRANSLATION__"
                        "REASONING_HEADROOM_TOKENS) if it emits reasoning tokens"
                    ),
                )
            else:
                reason = "every attempt returned nothing; source text kept as a placeholder"
                log.error("segment_never_translated", segment_id=segment_id, batch=batch.index)

        return SegmentOutcome(
            segment_id=segment_id,
            text=best.text,
            score=best.verdict.composite if best.verdict and best.verdict.score else None,
            needs_review=True,
            attempts=attempts,
            reason=reason,
        )

    # -- helpers ---------------------------------------------------------------------

    def _model_for(self, attempt_no: int) -> str:
        """Attempt 3 escalates to a stronger model (§11)."""
        if attempt_no >= self.settings.retry.max_attempts:
            return self.settings.models.role("escalation")
        return self.settings.models.translator

    def _build_retry_context(
        self,
        result: BatchTranslation,
        validation: ValidationReport,
        evaluation: BatchEvaluation | None,
    ) -> RetryContext:
        issues = [
            issue.model_dump()
            for verdict in (evaluation.verdicts.values() if evaluation else [])
            for issue in verdict.issues
        ]
        return RetryContext(
            previous_output=result.response.text[:6000],
            issues=issues,
            mechanical_problems=validation.mechanical_problems(),
        )

    def _index_units(self, batches: list[Batch]) -> None:
        """Build the unit ordering once per run, not once per batch.

        Re-segmenting the whole document to place each batch made context assembly O(n²) in
        the number of batches -- unnoticeable on a fixture, minutes of pure CPU on a
        400-page novel, and entirely wasted since the segmentation cannot change mid-run.
        """
        self._units = [unit for batch in batches for unit in batch.units]
        self._position = {unit.id: index for index, unit in enumerate(self._units)}

    def _context_for(self, batch: Batch) -> BatchContext:
        """Read-only context for a batch, from what is already translated."""
        if not self._units:
            self._index_units([batch])

        first = self._position.get(batch.ids[0], 0)
        last = self._position.get(batch.ids[-1], 0)
        look_back = self.settings.context.previous_target_blocks
        look_ahead = self.settings.context.next_source_blocks

        previous_ids = [u.id for u in self._units[max(0, first - look_back) : first]]
        return BatchContext(
            previous_target=[
                self.translations[pid] for pid in previous_ids if pid in self.translations
            ],
            next_source=[unit.text for unit in self._units[last + 1 : last + 1 + look_ahead]],
            rolling_summary=self.summaries.get(batch.chapter_id),
        )

    def _already_done(self) -> dict[str, str]:
        return {
            record.segment_id: record.final_text or ""
            for record in self.store.list_segments(self.job_id, status="done")
            if record.final_text
        }

    def _strip_done(self, batch: Batch, done: dict[str, str]) -> Batch | None:
        """Drop finished segments from a batch; return None if nothing is left."""
        if not done:
            return batch
        remaining = [unit for unit in batch.units if unit.id not in done]
        if not remaining:
            return None
        if len(remaining) == len(batch.units):
            return batch
        return Batch(
            index=batch.index,
            chapter_id=batch.chapter_id,
            units=remaining,
            oversized=batch.oversized,
        )


def seed_segments(store: JobStore, job_id: str, document: Document) -> int:
    """Write one ``segments`` row per translatable block, preserving finished work."""
    records = [
        SegmentRecord(
            segment_id=block.id,
            chapter_id=block.chapter_id,
            ordinal=index,
            kind=block.kind,
            source_text=block.text,
            final_text=None,
            final_score=None,
            status="pending",
            needs_review=False,
            attempts_count=0,
        )
        for index, block in enumerate(document.translatable_blocks())
    ]
    store.upsert_segments(job_id, records)
    return len(records)
