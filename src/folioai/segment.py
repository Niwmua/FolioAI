"""Segmentation: IR blocks into token-budgeted batches (brief §6).

Two rules that are never bent:

1. **A block is never split.** Translation units are whole blocks. Splitting a paragraph
   across two calls loses the sentence-level context that makes pronouns, tense and
   agreement come out right, and no amount of context-passing puts it back.
2. **A batch never crosses a chapter boundary.** Rolling context is per chapter, and a
   batch spanning two chapters would carry the wrong summary for half its content.

An oversized single block is sent alone with a raised per-call limit rather than being
split, which is the one case where the budget yields to rule 1.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .ir import Block, Document
from .logging_setup import get_logger
from .tags import render_segments
from .tokens import count_tokens

if TYPE_CHECKING:
    from .config import Settings

log = get_logger(__name__)


@dataclass(slots=True)
class Unit:
    """One translation unit: exactly one IR block."""

    block: Block
    tokens: int

    @property
    def id(self) -> str:
        return self.block.id

    @property
    def text(self) -> str:
        return self.block.text


@dataclass(slots=True, eq=False)
class Batch:
    """A set of units sent in one call, plus the context that travels with them.

    Identity-based equality (``eq=False``) so a batch can be used as a dict key while it is
    still being filled: two batches with the same units are still two different calls.
    """

    index: int
    chapter_id: str | None
    units: list[Unit] = field(default_factory=list)
    oversized: bool = False

    @property
    def ids(self) -> list[str]:
        return [unit.id for unit in self.units]

    @property
    def source_tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)

    @property
    def size(self) -> int:
        return len(self.units)

    def render(self) -> str:
        """The tagged user message for this batch."""
        return render_segments((unit.id, unit.text) for unit in self.units)

    def source_map(self) -> dict[str, str]:
        return {unit.id: unit.text for unit in self.units}


@dataclass(slots=True)
class BatchContext:
    """Read-only context for one batch (§6). Never translated, always labelled as context."""

    previous_target: list[str] = field(default_factory=list)
    next_source: list[str] = field(default_factory=list)
    rolling_summary: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.previous_target or self.next_source or self.rolling_summary)


def count_block_tokens(block: Block) -> int:
    """Token cost of one block's source text."""
    return count_tokens(block.text)


def build_units(doc: Document) -> list[Unit]:
    """Every translatable block, in document order, with its token count."""
    return [
        Unit(block=block, tokens=count_block_tokens(block)) for block in doc.translatable_blocks()
    ]


def build_batches(units: Sequence[Unit], settings: Settings) -> list[Batch]:
    """Group units into token-budgeted, chapter-bounded batches.

    Args:
        units: Translation units in document order.
        settings: Supplies ``translation.batch_tokens``.

    Returns:
        Batches in document order. Every unit appears in exactly one batch.
    """
    budget = settings.translation.batch_tokens
    batches: list[Batch] = []
    current: Batch | None = None

    for unit in units:
        # Rule 1: an oversized block travels alone rather than being cut in half.
        if unit.tokens > budget:
            if current is not None and current.units:
                batches.append(current)
                current = None
            batches.append(
                Batch(
                    index=len(batches),
                    chapter_id=unit.block.chapter_id,
                    units=[unit],
                    oversized=True,
                )
            )
            log.debug("oversized_block", block_id=unit.id, tokens=unit.tokens, budget=budget)
            continue

        crosses_chapter = current is not None and current.chapter_id != unit.block.chapter_id
        over_budget = current is not None and current.source_tokens + unit.tokens > budget

        if current is None or crosses_chapter or over_budget:
            if current is not None and current.units:
                batches.append(current)
            current = Batch(index=len(batches), chapter_id=unit.block.chapter_id)

        current.units.append(unit)

    if current is not None and current.units:
        batches.append(current)

    for position, batch in enumerate(batches):
        batch.index = position

    log.info(
        "segmentation_complete",
        units=len(units),
        batches=len(batches),
        budget_tokens=budget,
        oversized=sum(1 for b in batches if b.oversized),
    )
    return batches


def segment_document(doc: Document, settings: Settings) -> list[Batch]:
    """Convenience wrapper: IR straight to batches."""
    return build_batches(build_units(doc), settings)


def iter_batch_context(
    batches: Sequence[Batch],
    translations: dict[str, str],
    settings: Settings,
    summaries: dict[str, str] | None = None,
) -> Iterator[tuple[Batch, BatchContext]]:
    """Pair each batch with its read-only context.

    Previous *target* text comes from ``translations`` -- the model should see what it
    already produced, not the source it came from, or continuity of register is lost.
    """
    summaries = summaries or {}
    all_units = [unit for batch in batches for unit in batch.units]
    position_of = {unit.id: index for index, unit in enumerate(all_units)}

    for batch in batches:
        if not batch.units:
            continue
        first = position_of[batch.units[0].id]
        last = position_of[batch.units[-1].id]

        previous_ids = [
            u.id for u in all_units[max(0, first - settings.context.previous_target_blocks) : first]
        ]
        previous = [translations[pid] for pid in previous_ids if pid in translations]
        following = [
            unit.text
            for unit in all_units[last + 1 : last + 1 + settings.context.next_source_blocks]
        ]
        yield (
            batch,
            BatchContext(
                previous_target=previous,
                next_source=following,
                rolling_summary=summaries.get(batch.chapter_id or "", ""),
            ),
        )


def batch_statistics(batches: Sequence[Batch]) -> dict[str, int | float]:
    """Shape of a segmentation, for the estimate report and the logs."""
    if not batches:
        return {
            "batches": 0,
            "units": 0,
            "source_tokens": 0,
            "mean_units": 0.0,
            "max_tokens": 0,
            "oversized": 0,
        }
    sizes = [batch.size for batch in batches]
    tokens = [batch.source_tokens for batch in batches]
    return {
        "batches": len(batches),
        "units": sum(sizes),
        "source_tokens": sum(tokens),
        "mean_units": round(sum(sizes) / len(sizes), 1),
        "max_tokens": max(tokens),
        "oversized": sum(1 for batch in batches if batch.oversized),
    }
