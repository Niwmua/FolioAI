"""Cost projection before spending anything (brief §13).

``folioai estimate book.pdf --to de`` extracts and segments the book -- both free, both
offline -- and projects what a run would cost. It makes **zero** paid calls, which is also
what makes ``--dry-run`` honest.

The output is a *range*. A single number here would be false precision: expansion varies by
language pair, retry rate varies by model and source, and the estimate cannot know either
until the run is under way. A range that brackets reality is more useful than a point that
misses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table

from .ir import Document
from .llm.pricing import Cost, format_range, format_usd, price_call
from .logging_setup import get_logger
from .segment import Batch, batch_statistics, segment_document
from .tokens import count_tokens

if TYPE_CHECKING:
    from .config import Settings

log = get_logger(__name__)

#: Target tokens per source token, as (low, high). Ranges rather than points because the
#: real figure depends on the text as much as the language, and §9 learns the true ratio
#: from the running job anyway. German compounds but inflects; Chinese contracts sharply.
EXPANSION: dict[str, tuple[float, float]] = {
    "de": (1.05, 1.45),
    "nl": (1.05, 1.40),
    "fr": (1.10, 1.50),
    "es": (1.10, 1.50),
    "it": (1.05, 1.45),
    "pt": (1.10, 1.50),
    "ru": (0.95, 1.35),
    "pl": (0.95, 1.35),
    "ar": (0.90, 1.35),
    "fa": (0.95, 1.45),
    "ur": (0.95, 1.45),
    "he": (0.85, 1.25),
    "ja": (0.90, 1.60),
    "zh": (0.60, 1.10),
    "zh-hans": (0.60, 1.10),
    "zh-hant": (0.60, 1.10),
    "ko": (0.90, 1.50),
}
DEFAULT_EXPANSION = (0.85, 1.50)

#: Fixed prompt overhead per call: system prompt, style profile, glossary slice, context.
#: Measured from the shipped templates; an estimate that ignores it is short by ~30%.
TRANSLATE_OVERHEAD_TOKENS = 850
EVALUATE_OVERHEAD_TOKENS = 700

#: The judge reads a lot and writes a little: five scores and an issue list per segment.
EVAL_COMPLETION_TOKENS_PER_SEGMENT = 55

#: Retry rate multipliers applied to `budget.expected_retry_rate` for the low/high bounds.
RETRY_RANGE = (0.4, 2.0)


def expansion_range(target_lang: str) -> tuple[float, float]:
    """Target/source token ratio bounds for a language, falling back to a wide default."""
    key = target_lang.strip().lower()
    if key in EXPANSION:
        return EXPANSION[key]
    return EXPANSION.get(key.split("-")[0], DEFAULT_EXPANSION)


@dataclass(slots=True)
class PhaseEstimate:
    """Projected cost of one phase (translation, evaluation, retries) for one model."""

    name: str
    model: str
    calls: int
    prompt_tokens: int
    completion_low: int
    completion_high: int
    cost_low: float
    cost_high: float
    known_price: bool = True

    @property
    def cost_range(self) -> str:
        return format_range(self.cost_low, self.cost_high, known=self.known_price)


@dataclass(slots=True)
class Estimate:
    """The whole projection, ready to render or serialise."""

    source_path: str
    target_lang: str
    blocks: int
    translatable_blocks: int
    source_tokens: int
    words: int
    batches: int
    chapters: int
    phases: list[PhaseEstimate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_low(self) -> float:
        return sum(phase.cost_low for phase in self.phases)

    @property
    def cost_high(self) -> float:
        return sum(phase.cost_high for phase in self.phases)

    @property
    def all_prices_known(self) -> bool:
        return all(phase.known_price for phase in self.phases)

    @property
    def total_range(self) -> str:
        """The headline number. Stays short: the reason for an unknown is a warning line."""
        return format_range(self.cost_low, self.cost_high, known=self.all_prices_known)


def _phase(
    name: str,
    model: str,
    *,
    calls: int,
    prompt_tokens: int,
    completion_low: int,
    completion_high: int,
    settings: Settings,
) -> PhaseEstimate:
    low: Cost = price_call(model, prompt_tokens, completion_low, settings)
    high: Cost = price_call(model, prompt_tokens, completion_high, settings)
    return PhaseEstimate(
        name=name,
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_low=completion_low,
        completion_high=completion_high,
        cost_low=low.usd,
        cost_high=high.usd,
        known_price=low.known,
    )


def estimate_document(
    doc: Document,
    settings: Settings,
    *,
    target_lang: str,
    source_path: str | Path = "",
) -> Estimate:
    """Project the cost of translating an already-extracted document."""
    batches: list[Batch] = segment_document(doc, settings)
    stats = batch_statistics(batches)
    source_tokens = int(stats["source_tokens"])
    low_ratio, high_ratio = expansion_range(target_lang)

    estimate = Estimate(
        source_path=str(source_path),
        target_lang=target_lang,
        blocks=len(doc.blocks),
        translatable_blocks=len(doc.translatable_blocks()),
        source_tokens=source_tokens,
        words=sum(b.word_count for b in doc.translatable_blocks()),
        batches=len(batches),
        chapters=len(doc.chapters),
    )

    # -- translation -------------------------------------------------------------
    context_tokens = sum(
        count_tokens(unit.text)
        for batch in batches
        for unit in batch.units[: settings.context.previous_target_blocks]
    )
    translate_prompt = source_tokens + context_tokens + TRANSLATE_OVERHEAD_TOKENS * len(batches)
    translate_low = int(source_tokens * low_ratio)
    translate_high = int(source_tokens * high_ratio)
    estimate.phases.append(
        _phase(
            "translation",
            settings.models.translator,
            calls=len(batches),
            prompt_tokens=translate_prompt,
            completion_low=translate_low,
            completion_high=translate_high,
            settings=settings,
        )
    )

    # -- evaluation --------------------------------------------------------------
    sample = settings.evaluation.sample
    evaluated_segments = round(int(stats["units"]) * sample)
    if evaluated_segments:
        eval_batches = max(1, round(len(batches) * sample))
        # The judge sees source and target side by side, plus the rubric and glossary.
        eval_prompt = int(
            (source_tokens + translate_high) * sample + EVALUATE_OVERHEAD_TOKENS * eval_batches
        )
        eval_completion = evaluated_segments * EVAL_COMPLETION_TOKENS_PER_SEGMENT
        estimate.phases.append(
            _phase(
                "evaluation",
                settings.models.evaluator,
                calls=eval_batches,
                prompt_tokens=eval_prompt,
                completion_low=int(eval_completion * 0.7),
                completion_high=int(eval_completion * 1.4),
                settings=settings,
            )
        )
        if settings.evaluation.mode in {"back-translation", "both"}:
            share = 1.0 if settings.evaluation.mode == "back-translation" else 0.2
            estimate.phases.append(
                _phase(
                    "back-translation",
                    settings.models.role("back_translator"),
                    calls=max(1, int(eval_batches * share)),
                    prompt_tokens=int(translate_high * share),
                    completion_low=int(source_tokens * share * 0.8),
                    completion_high=int(source_tokens * share * 1.2),
                    settings=settings,
                )
            )

    # -- retries -----------------------------------------------------------------
    rate = settings.budget.expected_retry_rate
    low_rate, high_rate = rate * RETRY_RANGE[0], min(rate * RETRY_RANGE[1], 1.0)
    if high_rate > 0:
        # A retry re-sends the batch with the previous attempt and the issue list attached,
        # so its prompt is larger than the first attempt's, not the same size.
        retry_prompt = int(translate_prompt * high_rate * 1.6)
        estimate.phases.append(
            _phase(
                "retries",
                settings.models.translator,
                calls=round(len(batches) * high_rate),
                prompt_tokens=retry_prompt,
                completion_low=int(translate_low * low_rate),
                completion_high=int(translate_high * high_rate),
                settings=settings,
            )
        )

    if settings.models.translator == settings.models.evaluator:
        estimate.warnings.append(
            "Translator and evaluator are the same model. Two instances of one model share "
            "their blind spots, which is the main failure mode of LLM-as-judge (§10)."
        )
    if not estimate.all_prices_known:
        estimate.warnings.append(
            "Some models are missing from the pricing table, so their cost is counted as "
            "zero. Add them under 'pricing:' in your config for a real number."
        )
    if (
        settings.budget.max_cost_usd is not None
        and estimate.cost_high > settings.budget.max_cost_usd
    ):
        estimate.warnings.append(
            f"The high end of this estimate ({format_usd(estimate.cost_high)}) exceeds "
            f"--max-cost ({format_usd(settings.budget.max_cost_usd)}). The run will stop "
            "cleanly at the ceiling and stay resumable."
        )

    log.info(
        "estimate_complete",
        target=target_lang,
        blocks=estimate.blocks,
        batches=estimate.batches,
        source_tokens=source_tokens,
        cost_low=round(estimate.cost_low, 4),
        cost_high=round(estimate.cost_high, 4),
    )
    return estimate


def render_estimate(estimate: Estimate, settings: Settings) -> Table:
    """Render the projection.

    Five columns, not seven, and prompt/completion tokens share one cell: the table has to
    stay readable in an 80-column terminal, and a table where every cell is truncated to an
    ellipsis communicates nothing at all.
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        padding=(0, 1),
        title=f"{Path(estimate.source_path).name or 'document'} -> {estimate.target_lang}",
        title_style="heading",
        caption=(
            f"{estimate.words:,} words - {estimate.translatable_blocks:,} blocks - "
            f"{estimate.batches:,} calls - {estimate.chapters} chapters - "
            f"eval {settings.evaluation.sample:.0%}, "
            f"retries {settings.budget.expected_retry_rate:.0%}"
        ),
        caption_style="muted",
    )
    table.add_column("phase", no_wrap=True)
    table.add_column("model", no_wrap=True, overflow="ellipsis", max_width=20)
    table.add_column("calls", justify="right", no_wrap=True)
    table.add_column("tokens in/out", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)

    for phase in estimate.phases:
        table.add_row(
            phase.name,
            phase.model,
            f"{phase.calls:,}",
            f"{phase.prompt_tokens:,} / {phase.completion_low:,}-{phase.completion_high:,}",
            phase.cost_range,
        )
    table.add_section()
    table.add_row(
        "[heading]total[/heading]",
        "",
        "",
        "",
        f"[heading]{estimate.total_range}[/heading]",
    )
    return table
